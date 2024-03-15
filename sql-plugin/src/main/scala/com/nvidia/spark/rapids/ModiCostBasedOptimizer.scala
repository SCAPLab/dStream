/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import scala.collection.mutable.{ListBuffer, Map}

import ai.rapids.cudf._
import com.nvidia.spark.rapids.shims.{GlobalLimitShims, SparkShimImpl}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, Expression, GetStructField, WindowFrame, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.plans.{JoinType, LeftAnti, LeftSemi}
import org.apache.spark.sql.execution.{ExpandExec, FileSourceScanExec, FilterExec, GlobalLimitExec, LocalLimitExec, ProjectExec, SortExec, SparkPlan, SQLExecution, TakeOrderedAndProjectExec, UnionExec}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, StructType}



// *******************************************dStream******************************************
class Vertex(val node: SparkPlanMeta[SparkPlan], val version: String, val order: Int = 0) {
  var distance: Double = Double.MaxValue
  val ancestor: ListBuffer[Map[Vertex, Double]] = ListBuffer.empty[Map[Vertex, Double]]
  var pi: Vertex = null
}
// ********************************************************************************************

/**
 * Optimizer that can operate on a physical query plan.
 */
trait ModiOptimizer {

  /**
   * Apply optimizations to a query plan.
   *
   * @param conf Rapids configuration
   * @param plan The plan to optimize
   * @return A list of optimizations that were applied
   */
  def optimize(conf: RapidsConf, plan: SparkPlanMeta[SparkPlan]): Seq[Optimization]
}

/**
 * Experimental cost-based optimizer that aims to avoid moving sections of the plan to the GPU when
 * it would be better to keep that part of the plan on the CPU. For example, we don't want to move
 * data to the GPU just for a trivial projection and then have to move data back to the CPU on the
 * next step.
 */
class ModiCostBasedOptimizer(conf: RapidsConf) extends ModiOptimizer with Logging {

  /**
   * Walk the plan and determine CPU and GPU costs for each operator and then make decisions
   * about whether operators should run on CPU or GPU.
   *
   * @param conf Rapids configuration
   * @param plan The plan to optimize
   * @return A list of optimizations that were applied
   */

  // jgh: for LMStream
  // --------------------variables for LMStream--------------------------
  // the intention is to make the cost model pluggable since we are probably going to need to
  // experiment a fair bit with this part
  private val costModel = new DefaultCostModel(conf)

  // Declare private var to change currentBatchSize.
  // private var currentBatchSize: Double = currentBatchSizeOrigin
  // Since a single CPU core and GPU connection is in charge of one data partition,
  // we use currentBatchSize as the size of a single data partition.
  private var currentBatchSize: Double = _

  // Save max estimation of output size of each operator.
  private var previousOutputSize: Double = 0.0

  // jgh: originally fetched from spark.SparkPlan. but we don't know these initial value.
  // so, we set initial value to 150KB (as it says on the paper)
  private var inflectionPointOfBatchSize: Double = 150000.0

  // jgh: i think this value is number of dataset in Batch.
  private var sizeOfBatchedFileSource: Double = 0.0

  // jgh: size of Data Partition in MB
  private var numPartitions: Int = -1

  // jgh: Since sparkRapids allows 1:1=gpu:executor, this value is equal to number of executor.
  private val numTotalGpuDevices: Int = 1

  private var currentBatchSizeOrigin: Double = 0

  private var orderOfHashAgg: Int = -1
  private var orderOfProject: Int = -1
  private var orderOfFilter: Int = -1
  private var orderOfExpand: Int = -1
  private var orderOfJoin: Int = -1
  private var orderOfexchange: Int = -1
  private var orderOfSort: Int = -1

  private val startVertex: Vertex = new Vertex(null, "START", 0)
  private val sortedVertices: ListBuffer[Vertex] = ListBuffer.empty[Vertex]

  // to ensure the same instance that we used in DAG shortest path algorithm
  private val vertexMap: Map[(SparkPlanMeta[SparkPlan], String), Vertex] = Map()
  // Map to hold the other inputs of JOIN
  private val otherJoinInput: Map[SparkPlanMeta[SparkPlan], Vertex] = Map()
  //-----------------------------------------------------------------------

  def optimize(conf: RapidsConf, plan: SparkPlanMeta[SparkPlan]): Seq[Optimization] = {
    logTrace("CBO optimizing plan")
    val cpuCostModel = new ModiCpuCostModel(conf)
    val gpuCostModel = new ModiGpuCostModel(conf)
    val optimizations = new ListBuffer[Optimization]()

    // jgh start
    val currentMBSize = SparkPlan.getCurrentMBSize
    //    val totalMBSize = SparkPlan.getTotalMBSize
    val numDSOfMB = SparkPlan.getNumDSOfMB
    numPartitions = SparkPlan.getNumPartitions // for LMStream
    sizeOfBatchedFileSource = numDSOfMB // for LMStream
    currentBatchSizeOrigin = currentMBSize
    currentBatchSize = currentBatchSizeOrigin / numPartitions
    inflectionPointOfBatchSize = SparkPlan.getInflectionPoint match {
      case value if value < 1.0 => 150000.0
      case value => value
//      case _ => 150000.0
    }
    // jgh end

    // jgh
    //  recursivelyOptimize(conf, cpuCostModel, gpuCostModel, plan, optimizations,
    //    finalOperator = true)
    if (conf.lmOptimizerEnabled) {
      lmRecursivelyOptimize(conf, plan, optimizations, finalOperator = true)
      recursivelyOptimize(conf, cpuCostModel, gpuCostModel, plan, optimizations,
        finalOperator = true)
    } else {
      dStreamOptimize(conf, plan, optimizations) // jgh
    }

    if (optimizations.isEmpty) {
      logInfo(s"[Rapids_jgh] optimization is empty")
      logTrace(s"CBO finished optimizing plan. No optimizations applied.")
    } else {
      logInfo(s"[Rapids_jgh] optimization is not empty")
      // jgh: logTrace to logInfo
      logInfo(s"CBO finished optimizing plan. " +
        s"${optimizations.length} optimizations applied:\n\t${optimizations.mkString("\n\t")}")
    }
    optimizations
  }

  // *******************************************dStream******************************************
  def getOrCreateVertex(plan: SparkPlanMeta[SparkPlan], version: String, order: Int): Vertex = {
    if (!vertexMap.contains((plan, version))) {
      vertexMap((plan, version)) = new Vertex(plan, version, order)
    }
    vertexMap((plan, version))
  }

  def topologicalSort(plan: SparkPlanMeta[SparkPlan]): List[Vertex] = {
    val onlyCPUPlans = Set("EventTimeWatermarkExec", "StateStoreSaveExec",
    "StateStoreRestoreExec", "StreamingSymmetricHashJoinExec", "WriteToDataSourceV2Exec")


    def calculateTypeDim(childVersion: String, currentVersion: String): Int = {
      (childVersion, currentVersion) match {
        case ("CPU", "CPU") => 0
        case ("CPU", "GPU") => 2
        case ("GPU", "CPU") => 1
        case ("GPU", "GPU") => 3
      }
    }

    def calculateInputDim(inputSize: Double): Int = {
      inputSize match {
//        case a if a < 1024 => 0     // 1KB
//        case b if b < 5120 => 1     // 5KB
//        case c if c < 10240 => 2    // 10KB
//        case d if d < 51200 => 3    // 50KB
//        case e if e < 102400 => 4   // 100KB
//        case f if f < 512000 => 5   // 500KB
//        case g if g < 1048576 => 6  // 1MB
//        case h if h < 5242880 => 7  // 5MB
//        case i if i < 10485760 => 8 // 10MB
//        case _ => 9                 // 10MB ~
        case a if a < 1024 => 0    // 1KB
        case b if b < 5 * 1024 => 1    // 5KB
        case c if c < 10 * 1024 => 2    // 10KB
        case d if d < 20 * 1024 => 3    // 20KB
        case e if e < 30 * 1024 => 4    // 30KB
        case f if f < 40 * 1024 => 5    // 40KB
        case g if g < 50 * 1024 => 6    // 50KB
        case h if h < 60 * 1024 => 7    // 60KB
        case i if i < 70 * 1024 => 8    // 70KB
        case j if j < 80 * 1024 => 9    // 80KB
        case k if k < 90 * 1024 => 10    // 90KB
        case l if l < 100 * 1024 => 11    // 100KB
        case m if m < 200 * 1024 => 12    // 200KB
        case n if n < 300 * 1024 => 13    // 300KB
        case o if o < 400 * 1024 => 14    // 400KB
        case p if p < 500 * 1024 => 15    // 500KB
        case q if q < 600 * 1024 => 16    // 600KB
        case r if r < 700 * 1024 => 17    // 700KB
        case s if s < 800 * 1024 => 18    // 800KB
        case t if t < 900 * 1024 => 19    // 900KB
        case u if u < 1024 * 1024 => 20    // 1MB
        case v if v < 2 * 1024 * 1024 => 21    // 2MB
        case w if w < 3 * 1024 * 1024 => 22    // 3MB
        case x if x < 4 * 1024 * 1024 => 23    // 4MB
        case y if y < 5 * 1024 * 1024 => 24    // 5MB
        case z if z < 6 * 1024 * 1024 => 25    // 6MB
        case aa if aa < 7 * 1024 * 1024 => 26    // 7MB
        case bb if bb < 8 * 1024 * 1024 => 27    // 8MB
        case cc if cc < 9 * 1024 * 1024 => 28    // 9MB
        case _ => 29    // 9MB ~

      }
    }

    def getCost(plan: SparkPlan, typeDim: Int, inputDim: Int, order: Int): (Double, Double) = {
      val zeroDim = typeDim + order * 4
      plan.getClass.getSimpleName match {
        case filter if filter.contains("Filter") =>
          (SQLExecution.filterEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsFilterTable(zeroDim)(inputDim))
        case expand if expand.contains("Expand") =>
          (SQLExecution.expandEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsExpandTable(zeroDim)(inputDim))
        case project if project.contains("Project") =>
          (SQLExecution.projectEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsProjectTable(zeroDim)(inputDim))
        case join if join.contains("Join") =>
          (SQLExecution.hashJoinEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsHashJoinTable(zeroDim)(inputDim))
        case aggregate if aggregate.contains("Aggregate") =>
          (SQLExecution.hashAggregateEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsHashAggregateTable(zeroDim)(inputDim))
        case exchange if exchange.contains("Exchange") =>
          (SQLExecution.exchangeEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsExchangeTable(zeroDim)(inputDim))
        case sort if sort.contains("Sort") =>
          (SQLExecution.sortEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsSortTable(zeroDim)(inputDim))
        case stateSave if stateSave.contains("StateStoreSave") =>
          (SQLExecution.stateSaveEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsStateSaveTable(zeroDim)(inputDim))
        case stateRestore if stateRestore.contains("StateStoreRestore") =>
          (SQLExecution.stateRestoreEMATable(zeroDim)(inputDim),
            SQLExecution.estimateRowsStateRestoreTable(zeroDim)(inputDim))
        case _ => (1.0, currentBatchSizeOrigin / 78)
      }
    }

    def transferCost(inputSize: Double, typeDim: Int): Double = typeDim match {
      case 1 => inputSize * 0.0000078
      case 2 => inputSize * 0.0000078
      case _ => 0
    }

    if (plan.childPlans.isEmpty) {
//      plan.inputSize = currentBatchSizeOrigin
//      plan.outputSize = dStreamEstimateOutputSize(plan, currentBatchSizeOrigin)

      // TODO: need to consider the case where the plan is on GPU
      // select CPU or GPU via data size.
      if (currentBatchSizeOrigin > 1024) { // 50KB
        startVertex.ancestor += Map(getOrCreateVertex(plan, "GPU", 0) -> 0.0)
        val firstVertex = getOrCreateVertex(plan, "GPU", 0)
        sortedVertices.append(firstVertex)
        return firstVertex :: Nil
      } else {
        startVertex.ancestor += Map(getOrCreateVertex(plan, "CPU", 0) -> 0.0)
        val firstVertex = getOrCreateVertex(plan, "CPU", 0)
        sortedVertices.append(firstVertex)
        return firstVertex :: Nil
      }
    }

    val childList = plan.childPlans.flatMap(topologicalSort)

    val currOrder = plan.wrapped.getClass.getSimpleName match {
      case filter if filter.contains("Filter") => {
        orderOfFilter += 1
        orderOfFilter
      }
      case project if project.contains("Project") => {
        orderOfProject += 1
        orderOfProject
      }
      case aggregate if aggregate.contains("Aggregate") => {
        orderOfHashAgg += 1
        orderOfHashAgg
      }
      case sort if sort.contains("Sort") => {
        orderOfSort += 1
        orderOfSort
      }
      case join if join.contains("Join") => {
        orderOfJoin += 1
        orderOfJoin
      }
      case expand if expand.contains("Expand") => {
        orderOfExpand += 1
        orderOfExpand
      }
      case exchange if exchange.contains("exchange") => {
        orderOfexchange += 1
        orderOfexchange
      }
      case _ => 0
    }
    val currentVertices = if (onlyCPUPlans.contains(plan.wrapped.getClass.getSimpleName)) {
      List(getOrCreateVertex(plan, "CPU", currOrder))
    } else {
      List(getOrCreateVertex(plan, "CPU", currOrder),
        getOrCreateVertex(plan, "GPU", currOrder))
    }

//    plan.childPlans.foreach { childPlan =>
//      val childVertices = topologicalSort(childPlan)
//
//      for (childVertex <- childVertices) {
//        for (currentVertex <- currentVertices) {
//          val typeDim = calculateTypeDim(childVertex.version, currentVertex.version)
//          val inputDim = calculateInputDim(plan.inputSize)
//          val cost = getCost(plan.wrapped, typeDim, inputDim)
//          childVertex.ancestor += Map(currentVertex -> cost)
//        }
//      }
//    }
//
//    currentVertices

//    println(s"[${plan.wrapped.getClass.getSimpleName}]'s childList: ${childList}}")

//    plan.inputSize = plan.childPlans.map(_.outputSize).sum
//    plan.outputSize = dStreamEstimateOutputSize(plan, plan.inputSize)

    for (childVertex <- childList) {
      for (currentVertex <- currentVertices) {
        val typeDim = calculateTypeDim(childVertex.version, currentVertex.version)
        val inputDim = calculateInputDim(currentBatchSizeOrigin)
//        val cost = getCost(plan.wrapped, typeDim, inputDim, currOrder) +
//          transferCost(currentVertex.node.inputSize, typeDim)
        val (cost, rowNum) = getCost(plan.wrapped, typeDim, inputDim, currOrder)
        val transCost = transferCost(rowNum, typeDim)
        val totalCost = cost + transCost
//        println(s"[${childVertex.order}, ${childVertex.version}]" +
//          s"${childVertex.node.wrapped.getClass.getSimpleName} ---> " +
//          s"${totalCost}(${cost}, ${transCost}) & (${typeDim + (currOrder * 4)}, " +
//          s"${inputDim}) ---> " +
//          s"[${currentVertex.order}, ${currentVertex.version}]" +
//          s"${currentVertex.node.wrapped.getClass.getSimpleName}")
        childVertex.ancestor += Map(currentVertex -> totalCost)
      }
    }

    sortedVertices.appendAll(currentVertices)
    currentVertices
  }

  //  private def dStreamEstimateInputSize(plan: SparkPlanMeta[_],
  //                                       currentBatchSize: Double,
  //                                       prevOutputSize: Double): Double = {
  //    plan.wrapped match {
  //      case _: FileSourceScanExec =>
  //        currentBatchSize
  //      case _ =>
  //        prevOutputSize
  //    }
  //  }

//  private def dStreamEstimateOutputSize(plan: SparkPlanMeta[_],
//                                        currentBatchSize: Double): Double = {
//    plan.wrapped match {
//      case _: ExpandExec => // Stateless operation (Unary)
//        currentBatchSize
//      case _: FileSourceScanExec => // Stateless operation
//        currentBatchSize
//      case _: FilterExec => // Stateless operation
//        currentBatchSize
//      case _: ProjectExec => // Stateless operation
//        currentBatchSize
//      case _: BroadcastHashJoinExec => // Stateful operation
//        currentBatchSize * 2
//      case _: HashAggregateExec => // Stateful operation
//        currentBatchSize * 2
//      case _: ShuffleExchangeExec => // Stateful operation
//        currentBatchSize * 2
//      case _: SortExec => // Stateful operation
//        currentBatchSize * 2
//      case _ =>
//        currentBatchSize
//    }
//  }

  /**
   * this is the implementation of dStream optimizer
   * @param plan The plan to optimize
   * @param optimizations Accumulator to store the optimizations that are applied
   */
  private def dStreamOptimize(
                               conf: RapidsConf,
                               plan: SparkPlanMeta[SparkPlan],
                               optimizations: ListBuffer[Optimization]): Unit = {
    // topological sort
    // scalastyle:off println
    println(s"@@@@@@@@@ START DSTREAM OPTIMIZER @@@@@@@@@")
    val createGraphStartTime = System.nanoTime()
    println(s"currentBatchSizeOrigin: ${currentBatchSizeOrigin}")
    // Usage
    startVertex.distance = 0.0
//    val sortedVerties = topologicalSort(plan)
    topologicalSort(plan)
    sortedVertices.prepend(startVertex)
    val createGraphEndTime = System.nanoTime()

//    for ((i, idx) <- sortedVertices.zipWithIndex) {
//      if (idx == 0) {
//        println(s"startVertex, ${i.version}, ${i.distance}")
//      } else {
////        println(s"(${i})${i.node.wrapped.getClass.getSimpleName}, ${i.version}," +
////          s" ${i.distance} || ${i.node.inputSize} || ${i.node.outputSize}}")
//        println(s"[${i.order}]${i.node.wrapped.getClass.getSimpleName}, ${i.version}," +
//          s" ${i.distance}")
//
//      }
//      for (j <- i.ancestor){
//        j.foreach { case (v, w) =>
//          println(s"ancestor: (${v})${v.node.wrapped.getClass.getSimpleName}[${v.version}]: ${w}")
//        }
//      }
//    }

    println(s"========= CALCULATE SHORTEST PATH =========")
    val calculateStartTime = System.nanoTime()
    for (u <- sortedVertices) {
//      println(s"-----------------------------------------")
//      u.node match {
//        case null => println(s"u : [${u.version}]startVertex")
//        case _ => println(s"u : [${u.version}]${u.node.wrapped.getClass.getSimpleName}")
//      }

      // directed (u -> v)
      for (vMap <- u.ancestor) {
        vMap.foreach { case (v, w) =>
//          v.node match {
//            case null => println(s"v : [${v.version}]${v.node.wrapped.getClass.getSimpleName}," +
//              s" w: ${w}")
//            case _ => println(s"v : [${v.version}]${v.node.wrapped.getClass.getSimpleName}," +
//              s" w: ${w}")
//          }
//          println(s"v.distance: ${v.distance} VS cost: ${u.distance + w}(u.dist=${u.distance}" +
//            s" + w=${w})")

          // If v is JOIN, then we need to check whether the other input is already in the map
          if (v.node.wrapped.getClass.getSimpleName.contains("Join")) {
            if (otherJoinInput.contains(v.node)) {
              val currJoinInputVertex = otherJoinInput(v.node)
              if (currJoinInputVertex.distance > u.distance + w) {
                otherJoinInput -= v.node
                otherJoinInput(v.node) = u
              }
            } else {
              otherJoinInput(v.node) = u
            }
          }

          if (v.distance > u.distance + w) {
            v.distance = u.distance + w
            v.pi = u
//            v.node match {
//              case null => println(s"[${v.version}]startVertex " +
//                s"distance is updated to ${v.distance}")
//              case _ => println(s"[${v.version}]${v.node.wrapped.getClass.getSimpleName} " +
//                s"distance is updated to ${v.distance}")
//            }
          }
        }
      }
    }
    val calculateEndTime = System.nanoTime()

    println(s"========= APPLY SHORTEST PATH =========")
    val applyStartTime = System.nanoTime()
    applyShortestPath(sortedVertices.last, optimizations)
    if (otherJoinInput.nonEmpty){
      applyShortestPath(otherJoinInput.head._2, optimizations)
    }
    val applyEndTime = System.nanoTime()

    println(s"========= PRINT SHORTEST PATH =========")
//    println(s"sortedVertices.last: ${sortedVertices.last}")
//    showPiChain(sortedVertices.last)

    println(s"@@@@@@@@@ END DSTREAM OPTIMIZER @@@@@@@@@")
    println(s"@@@@@@@@@ DSTREAM OPTIMIZER SUMMARY @@@@@@@@@")
    val createTime = createGraphEndTime - createGraphStartTime
    val calculateTime = calculateEndTime - calculateStartTime
    val applyTime = applyEndTime - applyStartTime
    val totalTime = createTime + calculateTime + applyTime
    println(s"createGraphTime: ${(createGraphEndTime - createGraphStartTime) / 1000} us")
    println(s"calculateTime: ${(calculateEndTime - calculateStartTime) / 1000} us")
    println(s"applyTime: ${(applyEndTime - applyStartTime) / 1000} us")
    println(s"totalTime: ${totalTime / 1000} us")
    println(s"@@@@@@@@@ END DSTREAM OPTIMIZER SUMMARY @@@@@@@@@")

//    println(s"!!!!!FilterEMATable!!!!!")
//    SQLExecution.filterEMATable.foreach { row =>
//      println(row.mkString(" , "))
//    }
//    println(s"!!!!!estimateRowsFilterTable!!!!!")
//    SQLExecution.estimateRowsFilterTable.foreach { row =>
//      println(row.mkString(" , "))
//    }
//    println(s"!!!!!hashAggregateEMATable!!!!!")
//    SQLExecution.hashAggregateEMATable.foreach { row =>
//      println(row.mkString(" , "))
//    }
//    println(s"!!!!!estimateRowsHashAggregateTable!!!!!")
//    SQLExecution.estimateRowsHashAggregateTable.foreach { row =>
//      println(row.mkString(" , "))
//    }
//    println(s"!!!!!show Result!!!!!")
//    showPiChain(sortedVertices.last)
    // scalastyle: on println
  }
  def applyShortestPath(vertex: Vertex, optimizations: ListBuffer[Optimization]): Unit = {
    var currentVertex: Vertex = vertex
    while (currentVertex != null) {
      currentVertex.node match {
        case null =>
        case node => {
          if (currentVertex.version == "CPU" || currentBatchSizeOrigin < 600000) {
            //            println(s"${node.wrapped.getClass.getSimpleName} is running on CPU")
            optimizations.append(ModiAvoidTransition(node))
            node.costPreventsRunningOnGpu()
          }
          if (currentVertex.ancestor.nonEmpty && currentVertex.ancestor.head.nonEmpty) {
            if (currentVertex.ancestor.head.keys.head.node.wrapped.getClass
              .getSimpleName.contains("Join")) {
              if (otherJoinInput.contains(node)) {
                otherJoinInput -= node
              }
            }
          }
        }
      }

      currentVertex = currentVertex.pi
    }
  }

//  def showPiChain(vertex: Vertex): Unit = {
//    var currentVertex: Vertex = vertex
//    var idx = 0
//    while (currentVertex != null) {
//      currentVertex.node match {
//        case null => println(s"[${idx}]Vertex: startVertex, Version: ${currentVertex.version}")
//        case _ => println(s"[${idx}]Vertex: ${currentVertex.node.wrapped.getClass.getSimpleName}"+
//          s", Version: ${currentVertex.version}")
//      }
//
//      currentVertex = currentVertex.pi
//      idx += 1
//    }
//  }

  // *********************************************************************************************


  /**
   * Walk the plan and determine CPU and GPU costs for each operator and then make decisions
   * about whether operators should run on CPU or GPU.
   *
   * @param plan The plan to optimize
   * @param optimizations Accumulator to store the optimizations that are applied
   * @param finalOperator Is this the final (root) operator? We have special behavior for this
   *                      case because we need the final output to be on the CPU in row format
   * @return Tuple containing (cpuCost, gpuCost) for the specified plan and the subset of the
   *         tree beneath it that is a candidate for optimization.
   */
  private def recursivelyOptimize(
                                   conf: RapidsConf,
                                   cpuCostModel: ModiCostModel,
                                   gpuCostModel: ModiCostModel,
                                   plan: SparkPlanMeta[SparkPlan],
                                   optimizations: ListBuffer[Optimization],
                                   finalOperator: Boolean): (Double, Double) = {





    val childplan: Seq[String] = plan.childPlans.map(_.wrapped.getClass.getSimpleName) // jgh
    logInfo(s"[Rapids_jgh] recursiveOp wrap's name : ${plan.wrapped.getClass.getSimpleName} \n" +
      s"recursiveOp wrap's parent : ${plan.parent.map(_.wrapped.getClass.getSimpleName)} \n" +
      s"recursiveOp wrap's childPlans : ${childplan} \n" +
      s"recursiveOp wrap's num of childPlans : ${plan.childPlans.size} \n" +
      s"recursiveOp wrap's Class Type : ${plan.getClass.getSimpleName}") // jgh

    // get the CPU and GPU cost of this operator (excluding cost of children)
    val operatorCpuCost = cpuCostModel.getCost(plan)
    val operatorGpuCost = gpuCostModel.getCost(plan)

    // get the CPU and GPU cost of the child plan(s)
    val childCosts = plan.childPlans
      .map(child => recursivelyOptimize(
        conf,
        cpuCostModel,
        gpuCostModel,
        child,
        optimizations,
        finalOperator = false))
    val (childCpuCosts, childGpuCosts) = childCosts.unzip

    // calculate total (this operator + children)
    val totalCpuCost = operatorCpuCost + childCpuCosts.sum
    var totalGpuCost = operatorGpuCost + childGpuCosts.sum
    logCosts(plan, "Operator costs", operatorCpuCost, operatorGpuCost)
    logCosts(plan, "Operator + child costs", totalCpuCost, totalGpuCost)

    plan.estimatedOutputRows = ModiRowCountPlanVisitor.visit(plan)

    // determine how many transitions between CPU and GPU are taking place between
    // the child operators and this operator
    val numTransitions = plan.childPlans
      .count(canRunOnGpu(_) != canRunOnGpu(plan))
    logCosts(plan, s"numTransitions=$numTransitions", totalCpuCost, totalGpuCost)

    if (numTransitions > 0) {
      // there are transitions between CPU and GPU so we need to calculate the transition costs
      // and also make decisions based on those costs to see whether any parts of the plan would
      // have been better off just staying on the CPU

      // is this operator on the GPU?
      if (canRunOnGpu(plan)) {
        // at least one child is transitioning from CPU to GPU so we calculate the
        // transition costs
        val transitionCost = plan.childPlans
          .filterNot(canRunOnGpu)
          .map(transitionToGpuCost(conf, _)).sum

        // if the GPU cost including transition is more than the CPU cost then avoid this
        // transition and reset the GPU cost
        if (operatorGpuCost + transitionCost > operatorCpuCost && !isExchangeOp(plan)) {
          // avoid transition and keep this operator on CPU
          optimizations.append(ModiAvoidTransition(plan))
          plan.costPreventsRunningOnGpu()
          // reset GPU cost
          totalGpuCost = totalCpuCost
          logCosts(plan, s"Avoid transition to GPU", totalCpuCost, totalGpuCost)
        } else {
          // add transition cost to total GPU cost
          totalGpuCost += transitionCost
          logCosts(plan, s"transitionFromCpuCost=$transitionCost",
            totalCpuCost, totalGpuCost)
        }
      } else {
        // at least one child is transitioning from GPU to CPU so we evaluate each of this
        // child plans to see if it was worth running on GPU now that we have the cost of
        // transitioning back to CPU
        plan.childPlans.zip(childCosts).foreach {
          case (child, childCosts) =>
            val (childCpuCost, childGpuCost) = childCosts
            val transitionCost = transitionToCpuCost(conf, child)
            val childGpuTotal = childGpuCost + transitionCost
            if (canRunOnGpu(child) && !isExchangeOp(child)
              && childGpuTotal > childCpuCost) {
              // force this child plan back onto CPU
              optimizations.append(ModiReplaceSection(
                child, totalCpuCost, totalGpuCost))
              child.recursiveCostPreventsRunningOnGpu()
            }
        }

        // recalculate the transition costs because child plans may have changed
        val transitionCost = plan.childPlans
          .filter(canRunOnGpu)
          .map(transitionToCpuCost(conf, _)).sum
        totalGpuCost += transitionCost
        logCosts(plan, s"transitionFromGpuCost=$transitionCost",
          totalCpuCost, totalGpuCost)
      }
    }

    // special behavior if this is the final operator in the plan because we always have the
    // cost of going back to CPU at the end
    if (finalOperator && canRunOnGpu(plan)) {
      val transitionCost = transitionToCpuCost(conf, plan)
      totalGpuCost += transitionCost
      logCosts(plan, s"final operator, transitionFromGpuCost=$transitionCost",
        totalCpuCost, totalGpuCost)
    }

    // jgh start
    // for test [motivation3 before/after device]
    // scalastyle:off println
//    if (plan.wrapped.getClass.getSimpleName == "HashAggregateExec") {
//      numOfHashAgg += 1
//      println(s"numOfHashAgg: ${numOfHashAgg} on recursivelyOptimize in ModiCostBasedOptimizer")
//
//      if (numOfHashAgg == 1) {
//        println(s"move HashAggregate to CPU on recursivelyOptimize in ModiCostBasedOptimizer")
//        optimizations.append(ModiAvoidTransition(plan))
//        plan.costPreventsRunningOnGpu()
//      }
//    }

//    if (plan.wrapped.getClass.getSimpleName == "ProjectExec") {
//      numOfProject += 1
//      println(s"numOfProject: ${numOfProject} on recursivelyOptimize in ModiCostBasedOptimizer")
//
//      if (numOfProject == 2) {
//        println(s"move ProjectExec to CPU on recursivelyOptimize in ModiCostBasedOptimizer")
//        optimizations.append(ModiAvoidTransition(plan))
//        plan.costPreventsRunningOnGpu()
//      }
//    }

    //    if (plan.wrapped.getClass.getSimpleName == "FilterExec") {
    //      numOfFilter += 1
    //      println(s"numOfFilter: ${numOfFilter} on recursivelyOptimize in ModiCostBasedOptimizer")
    //
    //      if (numOfFilter == 3) {
    //        println(s"move Filter to CPU on recursivelyOptimize in ModiCostBasedOptimizer")
    //        optimizations.append(ModiAvoidTransition(plan))
    //        plan.costPreventsRunningOnGpu()
    //      }
    //    }
    // scalastyle:on println
    // jgh end

    if (totalGpuCost > totalCpuCost) {
      // we have reached a point where we have transitioned onto GPU for part of this
      // plan but with no benefit from doing so, so we want to undo this and go back to CPU
      if (canRunOnGpu(plan) && !isExchangeOp(plan)) {
        // this plan would have been on GPU so we move it and onto CPU and recurse down
        // until we reach a part of the plan that is already on CPU and then stop
        optimizations.append(ModiReplaceSection(plan, totalCpuCost, totalGpuCost))
        plan.recursiveCostPreventsRunningOnGpu()
        // reset the costs because this section of the plan was not moved to GPU
        totalGpuCost = totalCpuCost
        logCosts(plan, s"ModiReplaceSection: ${plan}", totalCpuCost, totalGpuCost)
      }
    }

    if (!canRunOnGpu(plan) || isExchangeOp(plan)) {
      // reset the costs because this section of the plan was not moved to GPU
      totalGpuCost = totalCpuCost
      logCosts(plan, s"Reset costs (not on GPU / exchange)", totalCpuCost, totalGpuCost)
    }

    logCosts(plan, "END", totalCpuCost, totalGpuCost)
    (totalCpuCost, totalGpuCost)
  }

  /**
   * Walk the plan and determine CPU and GPU costs for each operator and then make decisions
   * about whether operators should run on CPU or GPU.
   *
   * @param plan The plan to optimize
   * @param optimizations Accumulator to store the optimizations that are applied
   * @param finalOperator Is this the final (root) operator? We have special behavior for this
   *                      case because we need the final output to be on the CPU in row format
   * @return Tuple containing (cpuCost, gpuCost) for the specified plan and the subset of the
   *         tree beneath it that is a candidate for optimization.
   */
  private def lmRecursivelyOptimize(
                                     conf: RapidsConf,
                                     plan: SparkPlanMeta[SparkPlan],
                                     optimizations: ListBuffer[Optimization],
                                     finalOperator: Boolean): (Double, Double) = {

    // get the CPU and GPU cost of the child plan(s)
    val childCosts = plan.childPlans
      .map(child => lmRecursivelyOptimize(
        conf,
        child.asInstanceOf[SparkPlanMeta[SparkPlan]],
        optimizations,
        finalOperator = false))

    // logInfo("----------------------------------------------------------")
    // logInfo(s"[RAPIDS] Current operation: ${plan.wrapped.getClass.getSimpleName}")
    // logInfo(s"[RAPIDS] Is this operator on the GPU? (initial): ${plan.canThisBeReplaced}")

    /* plan.parent match {
        case Some(p) =>
            logInfo(s"[RAPIDS] parent: ${p.wrapped.getClass.getSimpleName}")
        case None =>
            logInfo("[RAPIDS] Root of plan")
    } */

    val (childCpuCosts, childGpuCosts) = childCosts.unzip

    // get the CPU and GPU cost of this operator (excluding cost of children)
    val (operatorCpuCost, operatorGpuCost) = costModel.applyCost(plan,
      currentBatchSize, inflectionPointOfBatchSize)

    // calculate total (this operator + children)
    val totalCpuCost = operatorCpuCost + childCpuCosts.sum
    var totalGpuCost = operatorGpuCost + childGpuCosts.sum

    /* logInfo(s"[RAPIDS] childCpuCosts: $childCpuCosts")
    logInfo(s"[RAPIDS] childGpuCosts: $childGpuCosts")
    logInfo(s"[RAPIDS] NormalizedBatchSize: ${(currentBatchSize / inflectionPointOfBatchSize)}")
    logInfo(s"[Rapids] operatorCpuCost: $operatorCpuCost")
    logInfo(s"[Rapids] operatorGpuCost: $operatorGpuCost")
    logInfo(s"[Rapids] totalCpuCost (before): $totalCpuCost")
    logInfo(s"[Rapids] totalGpuCost (before): $totalGpuCost") */

    plan.estimatedOutputRows = ModiRowCountPlanVisitor.visit(plan)

    // determine how many transitions between CPU and GPU are taking place between
    // the child operators and this operator
    val numTransitions = plan.childPlans
      .count(_.canThisBeReplaced != plan.canThisBeReplaced)

    if (numTransitions > 0) {
      // there are transitions between CPU and GPU so we need to calculate the transition costs
      // and also make decisions based on those costs to see whether any parts of the plan would
      // have been better off just staying on the CPU

      // is this operator on the GPU?
      if (plan.canThisBeReplaced) {
        // logInfo("[RAPIDS] This operator is on the GPU (numTransition > 0)")
        // at least one child is transitioning from CPU to GPU so we calculate the
        // transition costs
        val transitionCost = plan.childPlans.filter(!_.canThisBeReplaced)
          .map(lmTransitionToGpuCost(conf, _, currentBatchSize, inflectionPointOfBatchSize)).sum

        // if the GPU cost including transition is more than the CPU cost then avoid this
        // transition and reset the GPU cost
        if (operatorGpuCost + transitionCost > operatorCpuCost && !consumesQueryStage(plan)) {
          // avoid transition and keep this operator on CPU
          println(s"[RAPIDS] Operation ${plan.wrapped.getClass.getSimpleName}" +
            s"moved to CPU by cost model.")
          optimizations.append(ModiAvoidTransition(plan))
          plan.costPreventsRunningOnGpu()
          // reset GPU cost
          totalGpuCost = totalCpuCost;
        } else {
          // add transition cost to total GPU cost
          totalGpuCost += transitionCost
        }
      } else {
        // logInfo("[RAPIDS] This operator is not on the GPU (numTransition > 0)")
        // at least one child is transitioning from GPU to CPU so we evaulate each of this
        // child plans to see if it was worth running on GPU now that we have the cost of
        // transitioning back to CPU
        plan.childPlans.zip(childCosts).foreach {
          case (child, childCosts) =>
            val (childCpuCost, childGpuCost) = childCosts
            val transitionCost = lmTransitionToCpuCost(conf, child,
              currentBatchSize, inflectionPointOfBatchSize)
            val childGpuTotal = childGpuCost + transitionCost
            if (child.canThisBeReplaced && !consumesQueryStage(child)
              && childGpuTotal > childCpuCost) {
              println(s"[RAPIDS] Child operations of ${plan.wrapped.getClass.getSimpleName}" +
                s"moved to CPU by original cost model.")
              // force this child plan back onto CPU
              optimizations.append(ModiReplaceSection(
                child.asInstanceOf[SparkPlanMeta[SparkPlan]], totalCpuCost, totalGpuCost))
              child.recursiveCostPreventsRunningOnGpu()
            }
        }

        // recalculate the transition costs because child plans may have changed
        val transitionCost = plan.childPlans
          .filter(_.canThisBeReplaced)
          .map(lmTransitionToCpuCost(conf, _,
            currentBatchSize, inflectionPointOfBatchSize)).sum
        totalGpuCost += transitionCost
      }
    }

    // special behavior if this is the final operator in the plan because we always have the
    // cost of going back to CPU at the end
    if (finalOperator && plan.canThisBeReplaced) {
      // logInfo("[RAPIDS] Final operator behavior: Add cost of going back to the CPU.")
      totalGpuCost += lmTransitionToCpuCost(conf, plan,
        currentBatchSize, inflectionPointOfBatchSize)
    }

    // logInfo(s"[Rapids] totalCpuCost (last): $totalCpuCost")
    // logInfo(s"[Rapids] totalGpuCost (last): $totalGpuCost")

    if (totalGpuCost > totalCpuCost) {
      // println("[RAPIDS] totalGpuCost > totalCpuCost")
      // we have reached a point where we have transitioned onto GPU for part of this
      // plan but with no benefit from doing so, so we want to undo this and go back to CPU
      if (plan.canThisBeReplaced && !consumesQueryStage(plan)) {
        // this plan would have been on GPU so we move it and onto CPU and recurse down
        // until we reach a part of the plan that is already on CPU and then stop
        optimizations.append(ModiReplaceSection(plan, totalCpuCost, totalGpuCost))
        plan.recursiveCostPreventsRunningOnGpu()
        println(s"[RAPIDS] Operation ${plan.wrapped.getClass.getSimpleName}" +
          s"moved to CPU by cost model.")
        // reset the costs because this section of the plan was not moved to GPU
        totalGpuCost = totalCpuCost
      }
    }

    // Custom special behavior (1): Stream + Batch jobs.
    // Inform that this is batch source read & Set its size as current batch size.
    // If BroadcastHashJoinExec will be on the CPU, also put BroadcastExchangeExec on the CPU.
    // If BroadcastHashJoinExec will be on the GPU, also put BroadcastExchangeExec on the GPU.
    plan match {
      case p =>
        if (sizeOfBatchedFileSource != -1.0 &&
          p.wrapped.getClass.getSimpleName == "BroadcastExchangeExec") {
          p.parent match {
            case Some(pp) =>
              if (pp.wrapped.getClass.getSimpleName == "BroadcastHashJoinExec") {
                println("[RAPIDS] Start processing batch data source.")
                // currentBatchSize = sizeOfBatchedFileSource
                currentBatchSize = sizeOfBatchedFileSource / numPartitions
                val cudaMemInfo = Cuda.memGetInfo()
                // [TODO] Should we use free memory or need to consider allocation fraction?
                val gpuMemFree = cudaMemInfo.free
                // If currentBatchSize is too large, BroadcastHashJoinExec will be moved to CPU;
                // which means BroadcastExchangeExec must be on the CPU too.
                // [TODO] Get estimated input size of BroadcastHashJoinExec using function.
                if (currentBatchSize > (gpuMemFree * numTotalGpuDevices)) {
                  if (plan.canThisBeReplaced && !consumesQueryStage(plan)) {
                    // avoid transition and keep this operator on CPU
                    optimizations.append(ModiAvoidTransition(plan))
                    plan.costPreventsRunningOnGpu()
                    println(s"[RAPIDS] Operation ${plan.wrapped.getClass.getSimpleName}" +
                      s"moved to CPU to prevent GPU out-of-memory error.")
                    // reset GPU cost
                    totalGpuCost = totalCpuCost;
                  }
                }
              }
            case None =>
              logInfo("[RAPIDS] Root of plan")
          }
        }
    }

    // Custom special behavior (2): The broadcast must be on the same device.
    // If BroadcastExchangeExec moved to the CPU, force BraodcastHashJoinExec on the CPU.
    if (plan.wrapped.getClass.getSimpleName == "BroadcastHashJoinExec" &&
      plan.childPlans.head.wrapped.getClass.getSimpleName == "BroadcastExchangeExec" &&
      !plan.childPlans.head.canThisBeReplaced) {
      if (plan.canThisBeReplaced && !consumesQueryStage(plan)) {
        // avoid transition and keep this operator on CPU
        optimizations.append(ModiAvoidTransition(plan))
        plan.costPreventsRunningOnGpu()
        println(s"[RAPIDS] Both ${plan.wrapped.getClass.getSimpleName} and" +
          s"${plan.childPlans.head.wrapped.getClass.getSimpleName} run on the CPU.")
        // reset GPU cost
        totalGpuCost = totalCpuCost;
      }
    }

    // Custom special behavior (3): When there is running out of GPU device memory.
    // If it is the first operator (scan) or
    // current operator's child operator has been moved to the CPU,
    // and current operator is on the GPU,
    // then we need to check data size and GPU free memory to prevent GPU out-of-memory error.
    if ((plan.childPlans.size == 0 || !plan.childPlans.head.canThisBeReplaced) &&
      plan.canThisBeReplaced) {
      val maxInputSize = lmEstimateInputSize(conf, plan,
        currentBatchSize, previousOutputSize)
      val cudaMemInfo = Cuda.memGetInfo()
      // [TODO] Should we use free memory or need to consider allocation fraction?
      val gpuMemFree = cudaMemInfo.free
      // logInfo(s"[Rapids] maxInputSize: $maxInputSize")
      // logInfo(s"[Rapids] gpuDeviceNum: $numTotalGpuDevices")
      // logInfo(s"[Rapids] GPU free memory per 1 device: $gpuMemFree")

      if (maxInputSize > (gpuMemFree * numTotalGpuDevices)) {
        // If estmiated maximum size of input requires more memory than available GPU memory,
        // we should undo this and go back to CPU.
        // println("[Rapids] maxInputSize > gpuMemFree")
        if (plan.canThisBeReplaced && !consumesQueryStage(plan)) {
          // avoid transition and keep this operator on CPU
          optimizations.append(ModiAvoidTransition(plan))
          plan.costPreventsRunningOnGpu()
          println(s"[RAPIDS] Operation ${plan.wrapped.getClass.getSimpleName}" +
            s"moved to CPU to prevent GPU out-of-memory error.")
          // reset GPU cost
          totalGpuCost = totalCpuCost;
        }
      }
    }

    // Save max estimation of current opertor's output size
    // to use for estimation of next operator's input size.
    previousOutputSize = lmEstimateOutputSize(conf, plan,
      currentBatchSize, inflectionPointOfBatchSize)

    if (!plan.canThisBeReplaced || consumesQueryStage(plan)) {
      // reset the costs because this section of the plan was not moved to GPU
      totalGpuCost = totalCpuCost
    }

    // logInfo(s"[Rapids] totalCpuCost (final): $totalCpuCost")
    // logInfo(s"[Rapids] totalGpuCost (final): $totalGpuCost")
    // logInfo(s"[RAPIDS] Is this operator on the GPU? (final): ${plan.canThisBeReplaced}")

    (totalCpuCost, totalGpuCost)
  }

  private def logCosts(
                        plan: SparkPlanMeta[_],
                        message: String,
                        cpuCost: Double,
                        gpuCost: Double): Unit = {
    val sign = if (cpuCost == gpuCost) {
      "=="
    } else if (cpuCost < gpuCost) {
      "<"
    } else {
      ">"
    }
    logTrace(s"CBO [${plan.wrapped.getClass.getSimpleName}] $message: " +
      s"cpuCost=$cpuCost $sign gpuCost=$gpuCost)")
  }

  private def canRunOnGpu(plan: SparkPlanMeta[_]): Boolean = plan.wrapped match {
    case _: AdaptiveSparkPlanExec =>
      // this is hacky but AdaptiveSparkPlanExec is always tagged as "cannot replace" and
      // there are no child plans to inspect, so we just assume that the plan is running
      // on GPU
      true
    case _ => plan.canThisBeReplaced
  }

  private def transitionToGpuCost(conf: RapidsConf, plan: SparkPlanMeta[SparkPlan]): Double = {
    val rowCount = ModiRowCountPlanVisitor.visit(plan).map(_.toDouble)
      .getOrElse(conf.defaultRowCount.toDouble)
    val dataSize = ModiMemoryCostHelper.estimateGpuMemory(plan.wrapped.schema, rowCount)
    conf.getGpuOperatorCost("GpuRowToColumnarExec").getOrElse(0d) * rowCount +
      ModiMemoryCostHelper.calculateCost(dataSize, conf.cpuReadMemorySpeed) +
      ModiMemoryCostHelper.calculateCost(dataSize, conf.gpuWriteMemorySpeed)
  }

  private def transitionToCpuCost(conf: RapidsConf, plan: SparkPlanMeta[SparkPlan]): Double = {
    val rowCount = ModiRowCountPlanVisitor.visit(plan).map(_.toDouble)
      .getOrElse(conf.defaultRowCount.toDouble)
    val dataSize = ModiMemoryCostHelper.estimateGpuMemory(plan.wrapped.schema, rowCount)
    conf.getGpuOperatorCost("GpuColumnarToRowExec").getOrElse(0d) * rowCount +
      ModiMemoryCostHelper.calculateCost(dataSize, conf.gpuReadMemorySpeed) +
      ModiMemoryCostHelper.calculateCost(dataSize, conf.cpuWriteMemorySpeed)
  }

  // jgh: for LMStream
  private def lmTransitionToGpuCost(conf: RapidsConf,
                                    plan: SparkPlanMeta[_],
                                    currentBatchSize: Double,
                                    inflectionPointOfBatchSize: Double) = {
    // Original: this is a placeholder for now;
    // we would want to try and calculate the transition cost
    // based on the data types and size (if known)
    // conf.defaultTransitionToGpuCost // 0.1

    // We should normalize score by reflecting batch size.
    // If current batch size is large -> transitionToGpuCost gets bigger -> avoid GPUs

    // jgh: The defaultTransitionToGpuCost conf does not exist in the 23.06.0 version,
    // so set it to 0.1
    //    conf.defaultTransitionToGpuCost * (currentBatchSize / inflectionPointOfBatchSize)
    0.1 * (currentBatchSize / inflectionPointOfBatchSize)
  }

  // jgh: for LMStream
  private def lmTransitionToCpuCost(conf: RapidsConf,
                                    plan: SparkPlanMeta[_],
                                    currentBatchSize: Double,
                                    inflectionPointOfBatchSize: Double) = {
    // this is a placeholder for now - we would want to try and calculate the transition cost
    // based on the data types and size (if known)
    // conf.defaultTransitionToCpuCost // 0.1

    // We should normalize score by reflecting batch size.
    // If current batch size is large -> transitionToCpuCost gets bigger -> avoid CPUs

    // jgh: The defaultTransitionToCpuCost conf does not exist in the 23.06.0 version,
    // so set it to 0.1
    //    conf.defaultTransitionToCpuCost * (currentBatchSize / inflectionPointOfBatchSize)
    0.1 * (currentBatchSize / inflectionPointOfBatchSize)
  }

  // jgh: for LMStream
  /**
   * Determines whether the specified operator will read from a query stage.
   */
  private def consumesQueryStage(plan: SparkPlanMeta[_]): Boolean = {
    // if the child query stage already executed on GPU then we need to keep the
    // next operator on GPU in these cases

    // jgh: spark 3.2.3 doen't manage CustomShuffleReaderExec. so delete CustomShuffleReaderExec
    SQLConf.get.adaptiveExecutionEnabled && (plan.wrapped match {
      //      case _: CustomShuffleReaderExec
      case _: ShuffledHashJoinExec
           | _: BroadcastHashJoinExec
           | _: BroadcastNestedLoopJoinExec => true
      case _ => false
    })
  }

  // jgh: for LMStream
  private def lmEstimateInputSize(conf: RapidsConf,
                                  plan: SparkPlanMeta[_],
                                  currentBatchSize: Double,
                                  previousOutputSize: Double) = {
    // Return max estimation of input size for each operators;
    // If operator is scan, input size is same as currentBatchSize
    // else if for other operators, input size is same as output size of previous operators
    plan.wrapped match {
      case _: FileSourceScanExec =>
        currentBatchSize
      case _ =>
        previousOutputSize
    }
  }

  // jgh: for LMStream
  private def lmEstimateOutputSize(conf: RapidsConf,
                                   plan: SparkPlanMeta[_],
                                   currentBatchSize: Double,
                                   inflectionPointOfBatchSize: Double) = {
    // Return max estimation of output size for each operators;
    // stateless operators' max output size is same as input size
    // stateful operators' max output size is bounded by multiple of input size
    // windowing operators and other streaming-state operators (e.g., stateStoreSaveExec)
    // are fixed on CPU, since it store its incremental state infos in host memory;
    // which results too much transition cost.
    plan.wrapped match {
      case _: ExpandExec => // Stateless operation (Unary)
        currentBatchSize
      case _: FileSourceScanExec => // Stateless operation
        currentBatchSize
      case _: FilterExec => // Stateless operation
        currentBatchSize
      case _: ProjectExec => // Stateless operation
        currentBatchSize
      case _: BroadcastHashJoinExec => // Stateful operation
        currentBatchSize * 2
      case _: HashAggregateExec => // Stateful operation
        currentBatchSize * 2
      case _: ShuffleExchangeExec => // Stateful operation
        currentBatchSize * 2
      case _: SortExec => // Stateful operation
        currentBatchSize * 2
      case _ =>
        currentBatchSize
    }
  }

  /**
   * Determines whether the specified operator is an exchange, or will read from an
   * exchange / query stage. CBO needs to avoid moving these operators back onto
   * CPU because it could result in an invalid plan.
   */
  private def isExchangeOp(plan: SparkPlanMeta[_]): Boolean = {
    // if the child query stage already executed on GPU then we need to keep the
    // next operator on GPU in these cases
    SparkShimImpl.isExchangeOp(plan)
  }
}

/**
 * The cost model is behind a trait so that we can consider making this pluggable in the future
 * so that users can override the cost model to suit specific use cases.
 */
trait ModiCostModel {

  /**
   * Determine the CPU and GPU cost for an individual operator.
   * @param plan Operator
   * @return (cpuCost, gpuCost)
   */
  def getCost(plan: SparkPlanMeta[_]): Double

}

// jgh: for lmStream
/**
 * The cost model is behind a trait so that we can consider making this pluggable in the future
 * so that users can override the cost model to suit specific use cases.
 */
trait LMCostModel {

  /**
   * Determine the CPU and GPU cost for an individual operator.
   * @param plan Operator
   * @return (cpuCost, gpuCost)
   */
  def applyCost(plan: SparkPlanMeta[_],
                currentBatchSize: Double,
                inflectionPointOfBatchSize: Double): (Double, Double)

  /**
   * Determine the cost of transitioning data from CPU to GPU for a specific operator
   * @param plan Operator
   * @return Cost
   */
  /* def transitionToGpuCost(plan: SparkPlanMeta[_],
    currentBatchSize: Double,
    inflectionPointOfBatchSize: Double): Double */

  /**
   * Determine the cost of transitioning data from GPU to CPU for a specific operator
   */
  /* def transitionToCpuCost(plan: SparkPlanMeta[_],
    currentBatchSize: Double,
    inflectionPointOfBatchSize: Double): Double */
}

// jgh : Originally there was no with Logging
class ModiCpuCostModel(conf: RapidsConf) extends ModiCostModel with Logging{

  def getCost(plan: SparkPlanMeta[_]): Double = {

    val rowCount = ModiRowCountPlanVisitor.visit(plan).map(_.toDouble)
      .getOrElse(conf.defaultRowCount.toDouble)

    val operatorCost = plan.conf
      .getCpuOperatorCost(plan.wrapped.getClass.getSimpleName)
      .getOrElse(conf.defaultCpuOperatorCost) * rowCount

    val operatorName = plan.wrapped.getClass.getSimpleName
    val cost = plan.conf
      .getCpuOperatorCost(plan.wrapped.getClass.getSimpleName)
      .getOrElse(conf.defaultCpuOperatorCost)

    val exprEvalCost = plan.childExprs
      .map(expr => exprCost(expr.asInstanceOf[BaseExprMeta[Expression]], rowCount))
      .sum

    logInfo(s"[Rapids_jgh] CPUoperatorName(cost) * rowNum : $operatorName" + s"($cost)" +
      s"* $rowCount")
    logInfo(s"[Rapids_jgh] Total CPU cost jgh$operatorName = operatorCost + exprEvalCost : " +
      s"${operatorCost+exprEvalCost} = $operatorCost + $exprEvalCost")

    operatorCost + exprEvalCost
  }

  private def exprCost[INPUT <: Expression](expr: BaseExprMeta[INPUT], rowCount: Double): Double = {
    if (ModiMemoryCostHelper.isExcludedFromCost(expr)) {
      return 0d
    }

    val memoryReadCost = expr.wrapped match {
      case _: Alias =>
        // alias has no cost, we just evaluate the cost of the aliased expression
        exprCost(expr.childExprs.head.asInstanceOf[BaseExprMeta[Expression]], rowCount)

      case _: AttributeReference | _: GetStructField =>
        ModiMemoryCostHelper.calculateCost(ModiMemoryCostHelper.estimateGpuMemory(
          expr.typeMeta.dataType, nullable = false, rowCount), conf.cpuReadMemorySpeed)

      case _ =>
        expr.childExprs
          .map(e => exprCost(e.asInstanceOf[BaseExprMeta[Expression]], rowCount)).sum
    }

    // the output of evaluating the expression needs to be written out to rows
    val memoryWriteCost = ModiMemoryCostHelper.calculateCost(ModiMemoryCostHelper.estimateGpuMemory(
      expr.typeMeta.dataType, nullable = false, rowCount), conf.cpuWriteMemorySpeed)

    // optional additional per-row overhead of evaluating the expression
    val exprEvalCost = rowCount *
      expr.conf.getCpuExpressionCost(expr.getClass.getSimpleName)
        .getOrElse(conf.defaultCpuExpressionCost)

    exprEvalCost + memoryReadCost + memoryWriteCost
  }
}

class ModiGpuCostModel(conf: RapidsConf) extends ModiCostModel with Logging{

  def getCost(plan: SparkPlanMeta[_]): Double = {
    val rowCount = ModiRowCountPlanVisitor.visit(plan).map(_.toDouble)
      .getOrElse(conf.defaultRowCount.toDouble)

    val operatorCost = plan.conf
      .getGpuOperatorCost(plan.wrapped.getClass.getSimpleName)
      .getOrElse(conf.defaultGpuOperatorCost) * rowCount

    val exprEvalCost = plan.childExprs
      .map(expr => exprCost(expr.asInstanceOf[BaseExprMeta[Expression]], rowCount))
      .sum

    // jgh start
    val operatorName = plan.wrapped.getClass.getSimpleName
    logInfo(s"[Rapids_jgh] GPUoperatorName(cost) * rowNum : $operatorName" + s"($operatorCost)" +
      s"* $rowCount")
    logInfo(s"[Rapids_jgh] Total GPU cost jgh$operatorName = operatorCost + exprEvalCost : " +
      s"${operatorCost+exprEvalCost} = $operatorCost + $exprEvalCost")
    // jgh end

    operatorCost + exprEvalCost
  }

  private def exprCost[INPUT <: Expression](expr: BaseExprMeta[INPUT], rowCount: Double): Double = {
    if (ModiMemoryCostHelper.isExcludedFromCost(expr)) {
      return 0d
    }

    var memoryReadCost = 0d
    var memoryWriteCost = 0d

    expr.wrapped match {
      case _: Alias =>
        // alias has no cost, we just evaluate the cost of the aliased expression
        exprCost(expr.childExprs.head.asInstanceOf[BaseExprMeta[Expression]], rowCount)

      case _: AttributeReference =>
      // referencing an existing column on GPU is almost free since we're
      // just increasing a reference count and not actually copying any data

      case _ =>
        memoryReadCost = expr.childExprs
          .map(e => exprCost(e.asInstanceOf[BaseExprMeta[Expression]], rowCount)).sum

        memoryWriteCost += ModiMemoryCostHelper
          .calculateCost(ModiMemoryCostHelper.estimateGpuMemory(
            expr.typeMeta.dataType, nullable = false, rowCount), conf.gpuWriteMemorySpeed)
    }

    // optional additional per-row overhead of evaluating the expression
    val exprEvalCost = rowCount *
      expr.conf.getGpuExpressionCost(expr.getClass.getSimpleName)
        .getOrElse(conf.defaultGpuExpressionCost)

    exprEvalCost + memoryReadCost + memoryWriteCost
  }
}

// jgh: for LMStream
class DefaultCostModel(conf: RapidsConf) extends LMCostModel {

  /* def transitionToGpuCost(plan: SparkPlanMeta[_]) = {
    // this is a placeholder for now - we would want to try and calculate the transition cost
    // based on the data types and size (if known)
    conf.defaultTransitionToGpuCost
  } */

  /* def transitionToCpuCost(plan: SparkPlanMeta[_]) = {
    // this is a placeholder for now - we would want to try and calculate the transition cost
    // based on the data types and size (if known)
    conf.defaultTransitionToCpuCost
  } */

  override def applyCost(plan: SparkPlanMeta[_],
                         currentBatchSize: Double,
                         inflectionPointOfBatchSize: Double): (Double, Double) = {

    // This is used for normalizing scores by reflecting
    // current batch size and optimized inflection point.
    val normalizedBatchSize = currentBatchSize / inflectionPointOfBatchSize

    // for now we have a constant cost for CPU operations and we make the GPU cost relative
    // to this but later we may want to calculate actual CPU costs
    val cpuCost = 1.0
    // CPU cost is proportional to normalized batch size;
    // which means if current batch size is large -> normalized batch size is large ->
    // then CPU preference goes down -> so we shold make normalizedCpuCost bigger
    // to maintain GPU functions.
    val normalizedCpuCost = cpuCost * normalizedBatchSize

    // always check for user overrides first
    // jgh: In 23.06.0 version, there is no def getOperatorCost. so it change to 0.8.
    // (default GpuOperatorCost is 0.8 in Rapids 0.5 version)
    val gpuCost = plan.conf.getGpuOperatorCost(plan.wrapped.getClass.getSimpleName).getOrElse {
      plan.wrapped match {
        case _: HashAggregateExec =>
          // Prefers CPU
          1.0
        case _: FilterExec =>
          // Prefers CPU
          1.0
        case _: ProjectExec =>
          // Original code: the cost of a projection is the average cost of its expressions
          /* plan.childExprs
              .map(expr => exprCost(expr.asInstanceOf[BaseExprMeta[Expression]]))
              .sum / plan.childExprs.length */
          // Preference score is borderline of CPU and GPU
          0.9
        case _: ShuffleExchangeExec =>
          // setting the GPU cost of ShuffleExchangeExec to 1.0 avoids moving from CPU to GPU for
          // a shuffle. This must happen before the join consistency or we risk running into issues
          // with disabling one exchange that would make a join inconsistent
          // Addition: SuffleExchangeExec prefers CPU since it needs to flush intermediate data
          // from host memory to disk.
          1.0
        case _: BroadcastHashJoinExec =>
          // Preference score is borderline of CPU and GPU
          0.9
        case _: ExpandExec =>
          // Preference score is borderline of CPU and GPU
          0.9
        case _: SortExec =>
          // Prefers GPU
          0.8
        case _: FileSourceScanExec =>
          // Prefers GPU
          0.8
        case _ =>
          // Original code; 0.8
          // conf.defaultOperatorCost
          // For other operators, basically set to prefer GPU
          0.8
      }
    }
    // GPU cost is unproportional to normalized batch size;
    // which means if current batch size is large -> normalized batch size is large ->
    // then GPU preference goes up -> so we should make normalizedGpuCost smaller
    // to not to be replaced by CPU functions.
    val normalizedGpuCost = gpuCost * (1 / normalizedBatchSize)

    // plan.cpuCost = cpuCost
    // plan.gpuCost = gpuCost
    plan.cpuCost = normalizedCpuCost
    plan.gpuCost = normalizedGpuCost

    (normalizedCpuCost, normalizedGpuCost)
  }

  //  private def exprCost[INPUT <: Expression](expr: BaseExprMeta[INPUT]): Double = {
  //    // always check for user overrides first
  //
  //    // jgh: getExpressionCost doesn't exist in 23.06.0 version.
  //    // change getExpressionCost to getGpuExpressionCost
  //    expr.conf.getGpuExpressionCost(expr.getClass.getSimpleName).getOrElse {
  //      expr match {
  //        case cast: CastExprMeta[_] =>
  //          // different CAST operations have different costs, so we allow these to be configured
  //          // based on the data types involved
  //
  //          // jgh: defaultExpressionCost doesn't exist in 23.06.0 version.
  //          // set this value to 0.8
  //          expr.conf.getGpuExpressionCost(s"Cast${cast.fromType}To${cast.toType}")
  //            .getOrElse(0.8)
  //        case _ =>
  //          // many of our BaseExprMeta implementations are anonymous classes so we look directly
  //          // at the wrapped expressions in some cases
  //          expr.wrapped match {
  //            case _: AttributeReference => 1.0 // no benefit on GPU
  //            case Alias(_: AttributeReference, _) => 1.0 // no benefit on GPU
  //            case _ => 0.8
  //          }
  //      }
  //    }
  //  }

}

object ModiMemoryCostHelper {
  private val GIGABYTE = 1024d * 1024d * 1024d

  /**
   * Calculate the cost (time) of transferring data at a given memory speed.
   *
   * @param dataSize Size of data to transfer, in bytes.
   * @param memorySpeed Memory speed, in GB/s.
   * @return Time in seconds.
   */
  def calculateCost(dataSize: Long, memorySpeed: Double): Double = {
    (dataSize / GIGABYTE) / memorySpeed
  }

  def isExcludedFromCost[INPUT <: Expression](expr: BaseExprMeta[INPUT]) = {
    expr.wrapped match {
      case _: WindowSpecDefinition | _: WindowFrame =>
        // Window expressions are Unevaluable and accessing dataType causes an exception
        true
      case _ => false
    }
  }

  def estimateGpuMemory(schema: StructType, rowCount: Double): Long = {
    // cardinality estimates tend to grow to very large numbers with nested joins so
    // we apply a maximum to the row count that we use when estimating data sizes in
    // order to avoid integer overflow
    val safeRowCount = rowCount.min(Int.MaxValue).toLong
    GpuBatchUtils.estimateGpuMemory(schema, safeRowCount)
  }

  def estimateGpuMemory(dataType: Option[DataType], nullable: Boolean, rowCount: Double): Long = {
    dataType match {
      case Some(dt) =>
        // cardinality estimates tend to grow to very large numbers with nested joins so
        // we apply a maximum to the row count that we use when estimating data sizes in
        // order to avoid integer overflow
        val safeRowCount = rowCount.min(Int.MaxValue).toLong
        GpuBatchUtils.estimateGpuMemory(dt, nullable, safeRowCount)
      case None =>
        throw new UnsupportedOperationException("Data type is None")
    }
  }
}

/**
 * Estimate the number of rows that an operator will output. Note that these row counts are
 * the aggregate across all output partitions.
 *
 * Logic is based on Spark's SizeInBytesOnlyStatsPlanVisitor. which operates on logical plans
 * and only computes data sizes, not row counts.
 */
object ModiRowCountPlanVisitor {

  def visit(plan: SparkPlanMeta[_]): Option[BigInt] = plan.wrapped match {
    case p: QueryStageExec =>
      p.getRuntimeStatistics.rowCount
    case _: GlobalLimitExec =>
      GlobalLimitShims.visit(plan.asInstanceOf[SparkPlanMeta[GlobalLimitExec]])
    case LocalLimitExec(limit, _) =>
      // LocalLimit applies the same limit for each partition
      val n = limit * plan.wrapped.asInstanceOf[SparkPlan]
        .outputPartitioning.numPartitions
      visit(plan.childPlans.head).map(_.min(n)).orElse(Some(n))
    case p: TakeOrderedAndProjectExec =>
      visit(plan.childPlans.head).map(_.min(p.limit)).orElse(Some(p.limit))
    case p: HashAggregateExec if p.groupingExpressions.isEmpty =>
      Some(1)
    case p: SortMergeJoinExec =>
      estimateJoin(plan, p.joinType)
    case p: ShuffledHashJoinExec =>
      estimateJoin(plan, p.joinType)
    case p: BroadcastHashJoinExec =>
      estimateJoin(plan, p.joinType)
    case _: UnionExec =>
      Some(plan.childPlans.flatMap(visit).sum)
    case _ =>
      default(plan)
  }

  private def estimateJoin(plan: SparkPlanMeta[_], joinType: JoinType): Option[BigInt] = {
    joinType match {
      case LeftAnti | LeftSemi =>
        // LeftSemi and LeftAnti won't ever be bigger than left
        visit(plan.childPlans.head)
      case _ =>
        default(plan)
    }
  }

  /**
   * The default row count is the product of the row count of all child plans.
   */
  private def default(p: SparkPlanMeta[_]): Option[BigInt] = {
    val one = BigInt(1)
    val product = p.childPlans.map(visit)
      .filter(_.exists(_ > 0L))
      .map(_.get)
      .product
    if (product == one) {
      // product will be 1 when there are no child plans
      None
    } else {
      Some(product)
    }
  }
}

// jgh
// sealed abstract class Optimization
// abstract class Optimization

case class ModiAvoidTransition[INPUT <: SparkPlan](plan: SparkPlanMeta[INPUT])
  extends Optimization {
  override def toString: String = s"It is not worth moving to GPU for operator: " +
    s"${ModiExplain.format(plan)}"
}

case class ModiReplaceSection[INPUT <: SparkPlan](
                                                   plan: SparkPlanMeta[INPUT],
                                                   totalCpuCost: Double,
                                                   totalGpuCost: Double) extends Optimization {
  override def toString: String = s"It is not worth keeping this section on GPU; " +
    s"gpuCost=$totalGpuCost, cpuCost=$totalCpuCost:\n${ModiExplain.format(plan)}"
}

object ModiExplain {

  def format(plan: SparkPlanMeta[_]): String = {
    plan.wrapped match {
      case p: SparkPlan => p.simpleString(SQLConf.get.maxToStringFields)
      case other => other.toString
    }
  }

  def formatTree(plan: SparkPlanMeta[_]): String = {
    val b = new StringBuilder
    formatTree(plan, b, "")
    b.toString
  }

  def formatTree(plan: SparkPlanMeta[_], b: StringBuilder, indent: String): Unit = {
    b.append(indent)
    b.append(format(plan))
    b.append('\n')
    plan.childPlans.filter(_.canThisBeReplaced)
      .foreach(child => formatTree(child, b, indent + "  "))
  }

}
