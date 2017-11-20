package com.kyrie.start

import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.DenseDoubleVector
import com.tencent.angel.ml.model.{PSModel, MLModel}
import com.tencent.angel.ml.predict.PredictResult
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.hadoop.conf.Configuration

/**
 * Created by tend on 2017/11/20.
 * 1.定义逻辑回归模型
 */
class LRModel(_ctx:TaskContext,conf:Configuration) extends MLModel(conf,_ctx) {

  val N = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)

  val weight = PSModel[DenseDoubleVector]("qs.lr.weight", 1, N)
    .setAverage(true)
  addPSModel(weight) //添加一个N维的PSModel给LRModel

  setSavePath(conf) //保存model
  setLoadPath(conf) //加载model

  override def predict(storage: DataBlock[LabeledData]): DataBlock[PredictResult] = ???
}
