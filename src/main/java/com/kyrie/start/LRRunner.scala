package com.kyrie.start

import com.tencent.angel.ml.MLRunner
import com.tencent.angel.ml.classification.lr.LRModel
import org.apache.hadoop.conf.Configuration

/**
 * Created by tend on 2017/11/20.
 * 3.需要实现Runner类将训练这个模型的任务提交到集群。
 */
class LRRunner extends MLRunner {

  /**
   *  training job to obtrain a model
   */

  override def train(conf: Configuration): Unit = {
    train(conf,LRModel(conf),classOf[LRTrainTask])
  }

  override def incTrain(conf: Configuration): Unit = ???

  override def predict(conf: Configuration): Unit = ???
}
