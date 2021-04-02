package com.fzr.util.appCluster

import com.fzr.util.HBaseUtil
import com.fzr.util.appCluster.AppCluster.AppList
import com.fzr.util.constants.AppConstants
import kafka.serializer.StringDecoder
import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.util.{Bytes, MD5Hash}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.slf4j.{Logger, LoggerFactory}
import org.slf4j.event.Level


object AppClusterDstream {
	
	def main(args: Array[String]): Unit = {
		val logger = LoggerFactory.getLogger(this.getClass)
		
		val batchTime = args(0).toInt
		
		val sparkSession = SparkSession.builder()
			.appName("AppClusterDstream")
			.config("spark.some.config.option", "some-value")
			.getOrCreate()
		
		val ssc = new StreamingContext(sparkSession.sparkContext,Seconds(batchTime))
		
		val gaTopic = Set("ga_nginx_hello_chick")
		val brokers = AppConstants.KAFKA_BJ_ESC
		val kafkaParams = Map[String, String](
			"metadata.broker.list" -> brokers,
			"serializer.class" -> "kafka.serializer.StringEncoder",
			"group.id" -> "lijingzhe.AppClusterDstream")
		
		// 加载过滤词表
		val filter_vocabulary = sparkSession.sparkContext.textFile(AppConstants.FILTER_PATH).collect().mkString(",")
		println("过滤词表：" + filter_vocabulary)
		// 用户列表流
		val gaStream = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParams, gaTopic)
		
		gaStream.map(_._2).print(3)
		
		val appStream = gaStream.map(line =>{
			if(StringUtils.isNoneBlank(line._2)){
				val uid = line._2.split(";")(0).toLong
				val appList = line._2.split(";")(1)
				(uid,appList)
			}else{
				(0l, "")
			}
		})
		
		val appLoadStream = appStream.filter(f=>f._1!=0l && StringUtils.isNotBlank(f._2) )
		
		appLoadStream.print(5)
		
		// 加载特征处理model
		val pips = PipelineModel.load(AppConstants.PIPES_PATH)
		// 加载K-Means model
		val kmm = KMeansModel
			.load(AppConstants.CLUSTER_PATH)
			.setFeaturesCol("app_list_idf")
			.setPredictionCol("prediction")
		
		appLoadStream.foreachRDD(rdd => {
			import sparkSession.implicits._
			// 如果在sparkstreaming中作外部变量broardcast或者累加器，需单例模式
			// val filter_vocabulary = FilterVocabularyInstance.getInstance(rdd.sparkContext)
			val df = rdd.map(f=>AppList(f._1,f._2.split(",").filter(filter_vocabulary.contains(_)))).toDF("uid",
				"app_list")
			val result = pips.transform(df).select("uid","app_list_idf")
			val kc = kmm.transform(result).select("uid","prediction")
			
			kc.rdd.map(row=>{
				val uid = row.getAs[Long]("uid").toString
				val rowKey = MD5Hash.getMD5AsHex(Bytes.toBytes(uid)).substring(0,8) + "_" + uid
				val prediction = row.getAs[Int]("prediction").toString
				val cols = Array(prediction)
				(new ImmutableBytesWritable, HBaseUtil.getPutAction(rowKey, "p", Array("pr"), cols))
			}).saveAsHadoopDataset(HBaseUtil.getJobConf("user_app_cluster"))
		})
		
		ssc.start()
		ssc.awaitTermination()
	}
}
