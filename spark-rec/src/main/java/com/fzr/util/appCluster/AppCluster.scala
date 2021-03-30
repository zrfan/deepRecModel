package com.fzr.util.appCluster


import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

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
		
		val spark = SparkSession.builder()
			.appName("AppCluster")
			.master("local[2]")
			.config("spark.some.config.option", "some-value")
			.getOrCreate()
		val conf = new Configuration()
		// 防止伪分布式出现错误：java.lang.IllegalArgumentException: Wrong FS: hdfs://lee:8020/appCluster/opzK, expected
		conf.set("fs.defaultFS", "hdfs://lee:8020")
		val fs = FileSystem.get(conf)
		val out = if(fs.exists(new Path(Constants.OPZK))){
			fs.append(new Path(Constants.OPZK))
		}else {
			fs.create(new Path(Constants.OPZK))
		}
		
	}
}
class AppCluster {
	
}
