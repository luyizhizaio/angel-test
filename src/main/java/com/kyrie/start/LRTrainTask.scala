package com.kyrie.start

import com.tencent.angel.ml.conf.MLConf._
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.{DenseDoubleVector, TDoubleVector}
import com.tencent.angel.ml.optimizer.sgd.L2LogLoss
import com.tencent.angel.ml.utils.DataParser
import com.tencent.angel.worker.task.{TrainTask, TaskContext}
import org.apache.commons.logging.{LogFactory, Log}
import org.apache.hadoop.io.{Text, LongWritable}

/**
 * Created by tend on 2017/11/20.
 * 2.Angel的模型的训练是在task中完成，
 * */
class LRTrainTask(val ctx:TaskContext) extends TrainTask[LongWritable,Text](ctx) {


  val LOG: Log = LogFactory.getLog(classOf[LRTrainTask])

  val epochNum = conf.getInt(ML_EPOCH_NUM, DEFAULT_ML_EPOCH_NUM)
  val lr = conf.getDouble(ML_LEARN_RATE, DEFAULT_ML_LEAR_RATE)
  val feaNum = conf.getInt(ML_FEATURE_NUM, DEFAULT_ML_FEATURE_NUM)
  val dataFmt = conf.get(ML_DATAFORMAT, DEFAULT_ML_DATAFORMAT)
  val reg = conf.getDouble(ML_REG_L2, DEFAULT_ML_REG_L2)
  val loss = new L2LogLoss(reg)


  override def train(ctx: TaskContext): Unit = {
    //逻辑回归模型
    val model = new LRModel(ctx,conf)

    while(ctx.getEpoch < epochNum){
      val weight = model.weight.getRow(0)
      val grad = batchGradientDescent(weight)
      // Get model and calculate gradient
      model.weight.increment(grad.timesBy(-1.0 * lr))
      model.weight.syncClock()

      // Increase iteration number
      ctx.incEpoch()
    }

  }
  //解析
  override def parse(key: LongWritable, value: Text): LabeledData = {
    DataParser.parseVector(key,value,feaNum,dataFmt,negY=true)
  }



  def batchGradientDescent(weight:TDoubleVector): TDoubleVector={
    var(grad,batchLoss) = (new DenseDoubleVector(feaNum),0.0)

    trainDataBlock.resetReadIndex()
    for(i <- 0 until trainDataBlock.size()){
      val data = trainDataBlock.read()
      val pred = weight.dot(data.getX)
      grad.plusBy(data.getX,-1.0 * loss.grad(pred,data.getY))
      batchLoss +=loss.loss(pred,data.getY)
    }
    grad.timesBy((1+reg)* 1.0 / trainDataBlock.size())
    LOG.info(s"Gradient descent batch ${ctx.getEpoch}, batch loss=$batchLoss")
    grad.setRowId(0)
    grad
  }
}
