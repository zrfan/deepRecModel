package com.fzr.util.appCluster

import com.fzr.util.HBaseUtil
import com.fzr.util.appCluster.AppCluster.AppList
import com.fzr.util.constants.AppConstants
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.util.{Bytes, MD5Hash}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

class ClusterPrediction {
	
}
object ClusterPrediction{
	def main(args: Array[String]): Unit = {
		val logger = LoggerFactory.getLogger(this.getClass)
		
		var numPartitions=10
		if(args.length>0) numPartitions=args(0).toInt
		
		val spark = SparkSession.builder()
			.appName("OnLineCluster")
			.config("spark.some.config.option", "some-value")
			.getOrCreate()
		
		import spark.implicits._
		// 加载textFile
		val filter_vocabulary = spark.sparkContext.textFile(AppConstants.FILTER_PATH).collect().mkString(",")
		println("过滤词表：" + filter_vocabulary)
		val df = spark.read.textFile(AppConstants.ONLINE_PATH ).map(row=>{
			val rows = row.split(",")
			AppList(rows(0).toLong,rows(1).split(";").filter(!filter_vocabulary.split(",").contains(_)))
		}).toDF("uid","app_list").repartition(numPartitions)
		
		// app_list列转化为稀疏特征向量
		val pips = PipelineModel.load(AppConstants.PIPES_PATH)
		val result = pips.transform(df).select("uid","app_list_idf")
		
		val kmm = KMeansModel
			.load(AppConstants.CLUSTER_PATH)
			.setFeaturesCol("app_list_idf")
			.setPredictionCol("prediction")
		val kc = kmm.transform(result).select("uid","prediction")
		
		val rdd1 = kc.rdd
		rdd1.saveAsTextFile(AppConstants.PREDICTION_PATH )
		
		rdd1.map(row=>{
			val uid = row.getAs[Long]("uid").toString
			val rowKey = MD5Hash.getMD5AsHex(Bytes.toBytes(uid)).substring(0,8) + "_" + uid
			val prediction = row.getAs[Int]("prediction").toString
			val cols = Array(prediction)
			(new ImmutableBytesWritable, HBaseUtil.getPutAction(rowKey, "p", Array("appLTag"), cols))
		}).saveAsHadoopDataset(HBaseUtil.getJobConf("user_app_cluster"))
		
		spark.stop()
	}
}
