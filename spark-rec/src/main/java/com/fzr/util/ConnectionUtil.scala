package com.fzr.util

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

public class ConnectionUtil {
	def getAllSparkSession(appName: String): SparkSession = {
		val sparkSession = SparkSession.builder.appName(appName)
			.config("spark.sql.warehouse.dir", "viewfs://c9/user_ext//warehouse")
			.config("hive.exec.scratchdir", "viewfs://c9/user_ext//wurui1/hive")
			.enableHiveSupport.getOrCreate
		sparkSession
	}
	
	def getAllSparkSession(appName: String, sparkConf: SparkConf): SparkSession = {
		val sparkSession = SparkSession.builder.appName(appName)
			.config("spark.sql.warehouse.dir", "viewfs://c9/user_ext//warehouse")
			.config("hive.exec.scratchdir", "viewfs://c9/user_ext//wurui1/hive")
			.config(sparkConf).enableHiveSupport.getOrCreate
		sparkSession
	}
}
