package com.fzr.util.appCluster


import com.fzr.util.constants.AppConstants
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.HashMap

/**
 * app 列表聚类
 */
object AppCluster{
	def main(args: Array[String]) ={
		Logger.getLogger("org").setLevel(Level.ERROR)
		var numPartitions=2
		if(args.length>0) numPartitions=args(0).toInt
		var maxIter = 20
		if(args.length>1) maxIter=args(1).toInt
		var minSeq=128
		if(args.length>2) minSeq=args(2).toInt
		var maxSeq=129
		if(args.length>3) maxSeq=args(3).toInt
		
		val sparkSession = SparkSession.builder()
			.appName("AppCluster")
			.master("local[2]")
			.config("spark.some.config.option", "some-value")
			.getOrCreate()
		val conf = new Configuration()
		// 防止伪分布式出现错误：java.lang.IllegalArgumentException: Wrong FS: hdfs://lee:8020/appCluster/opzK, expected
		conf.set("fs.defaultFS", "hdfs://lee:8020")
		val fs = FileSystem.get(conf)
		val out = if(fs.exists(new Path(AppConstants.OPZK_PATH))){
			fs.append(new Path(AppConstants.OPZK_PATH))
		}else {
			fs.create(new Path(AppConstants.OPZK_PATH))
		}
		
		val (result, map) = processCluster(sparkSession, numPartitions, minSeq, maxSeq, maxIter, fs)
		
		// 释放缓存
		result.unpersist()
		out.write(map.toString.getBytes("UTF-8"))
		sparkSession.stop()
	}
	def processCluster(sparkSession: SparkSession, numPartitions: Int, minSeq: Int, maxSeq: Int, maxIter: Int,
	                   fs: FileSystem): (DataFrame, HashMap[Int, Double]) ={
		import sparkSession.implicits._
		// 加载textFile
		val filter_vocabulary = sparkSession.sparkContext.textFile(AppConstants.FILTER_PATH).collect().mkString(",")
		println("过滤词表：" + filter_vocabulary)
		val df = sparkSession.read.textFile(AppConstants.SAMPLE_PATH).map(row=>{
			val rows = row.split(",")
			AppList(rows(0).toLong,rows(1).split(";").filter(!filter_vocabulary.split(",").contains(_)))
		}).toDF("uid","app_list").repartition(numPartitions)
		df.show(10,false)
		
		// app_list 列转化为稀疏特征向量
		val pips = if(!fs.exists(new Path(AppConstants.PIPES_PATH))){
			val htf = new HashingTF().setInputCol("app_list").setOutputCol("app_list_htf").setNumFeatures(math.pow(2,18).toInt)
			val idf = new IDF().setInputCol("app_list_htf").setOutputCol("app_list_idf")
			val p = new Pipeline().setStages(Array(htf,idf)).fit(df)
			p.write.overwrite().save(AppConstants.PIPES_PATH)
			p
		}else{
			PipelineModel.load(AppConstants.PIPES_PATH)
		}
		val result = pips.transform(df).select("uid","app_list_idf")
		result.cache()
		result.show(10,false)
		
		// result在Kmeans迭代寻优多次用到，因此加载内存
		val map = new HashMap[Int,Double]()
		// k-means聚类
		val ks = Range(minSeq, maxSeq)
		var minSsd = Double.MaxValue
		ks.foreach(cluster => {
			val kmeans = new KMeans()
				.setK(cluster)
				.setMaxIter(maxIter)
				.setFeaturesCol("app_list_idf")
				.setPredictionCol("prediction")
			val kmm = kmeans.fit(result)
			val ssd = kmm.computeCost(result)
			map.put(cluster, ssd)
			minSsd = if(ssd < minSsd){
				kmm.write.overwrite().save(AppConstants.CLUSTER_PATH)
				ssd
			}else minSsd
			
			println("当k=" + cluster + "，点到最近中心距离的平方和:"+ ssd)
		})
		(result, map)
	}
	case class AppList(uid: Long, app_list: Array[String]) extends Serializable
}
class AppCluster {
	
}
