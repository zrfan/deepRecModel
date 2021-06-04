package com.fzr.util


import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.client.BufferedMutator.ExceptionListener
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{BufferedMutator, BufferedMutatorParams, Connection, ConnectionFactory, Get, Mutation, Put, Result, RetriesExhaustedWithDetailsException, Scan}
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.hadoop.mapred.JobConf
import org.slf4j.LoggerFactory

object HBaseUtil {
	private val LOGGER = LoggerFactory.getLogger(this.getClass)
	private var connection: Connection = null
	private var conf: Configuration = null
	
	
	def init() = {
		conf = HBaseConfiguration.create()
		conf.set("hbase.zookeeper.quorum", "test")
		conf.set("zookeeper.znode.parent", "/hbase")
		connection = ConnectionFactory.createConnection(conf)
	}
	
	object HBaseClient extends Serializable {
		conf = HBaseConfiguration.create()
		conf.set("hbase.zookeeper.quorum", "test")
		conf.set("zookeeper.znode.parent", "/hbase")
		connection = ConnectionFactory.createConnection(conf)
		lazy val hook = new Thread {
			override def run = {
				connection.close()
			}
		}
	}
	
	
	def getJobConf(tableName: String) = {
		conf = HBaseConfiguration.create()
		val jobConf = new JobConf(conf)
		jobConf.set("hbase.zookeeper.quorum", "test")
		jobConf.set("zookeeper.znode.parent", "/hbase")
		jobConf.set("hbase.zookeeper.property.clientPort", "2181")
		jobConf.setOutputFormat(classOf[org.apache.hadoop.hbase.mapred.TableOutputFormat])
		jobConf.set(org.apache.hadoop.hbase.mapred.TableOutputFormat.OUTPUT_TABLE, tableName)
		jobConf
		
	}
	
	
	def getNewConf(tableName: String) = {
		conf = HBaseConfiguration.create()
		conf.set("hbase.zookeeper.quorum", "test")
		conf.set("zookeeper.znode.parent", "/hbase")
		conf.set("hbase.zookeeper.property.clientPort", "2181")
		conf.set(org.apache.hadoop.hbase.mapreduce.TableInputFormat.INPUT_TABLE, tableName)
		val scan = new Scan()
		conf.set(org.apache.hadoop.hbase.mapreduce.TableInputFormat.SCAN, Base64.encodeBytes(ProtobufUtil.toScan(scan).toByteArray))
		conf
	}
	
	
	def getNewJobConf(tableName: String) = {
		
		val conf = HBaseConfiguration.create()
		conf.set("hbase.zookeeper.quorum", "test")
		conf.set("zookeeper.znode.parent", "/hbase")
		conf.set("hbase.zookeeper.property.clientPort", "2181")
		conf.set("hbase.defaults.for.version.skip", "true")
		conf.set(org.apache.hadoop.hbase.mapreduce.TableOutputFormat.OUTPUT_TABLE, tableName)
		conf.setClass("mapreduce.job.outputformat.class", classOf[org.apache.hadoop.hbase.mapreduce.TableOutputFormat[String]],
			classOf[org.apache.hadoop.mapreduce.OutputFormat[String, Mutation]])
		new JobConf(conf)
	}
	
	
	def closeConnection(): Unit = {
		connection.close()
	}
	
	def getGetAction(rowKey: String): Get = {
		val getAction = new Get(Bytes.toBytes(rowKey));
		getAction.setCacheBlocks(false);
		getAction
	}
	
	
	def getPutAction(rowKey: String, familyName: String, column: Array[String], value: Array[String]): Put = {
		val put: Put = new Put(Bytes.toBytes(rowKey));
		for (i <- 0 until (column.length)) {
			put.add(Bytes.toBytes(familyName), Bytes.toBytes(column(i)), Bytes.toBytes(value(i)));
		}
		put
	}
	
	
	def insertData(tableName: String, put: Put) = {
		val name = TableName.valueOf(tableName)
		val table = connection.getTable(name)
		table.put(put)
	}
	
	
	def addDataBatchEx(tableName: String, puts: java.util.List[Put]): Unit = {
		
		val name = TableName.valueOf(tableName)
		val table = connection.getTable(name)
		val listener = new ExceptionListener {
			override def onException(e: RetriesExhaustedWithDetailsException, bufferedMutator: BufferedMutator): Unit =
			{
				for (i <- 0 until e.getNumExceptions) {
					LOGGER.info("写入put失败:" + e.getRow(i))
				}
			}
		}
		
		val params = new BufferedMutatorParams(name)
			.listener(listener)
			.writeBufferSize(4 * 1024 * 1024)
		try {
			val mutator = connection.getBufferedMutator(params)
			mutator.mutate(puts)
			mutator.close()
		} catch {
			case e: Throwable => e.printStackTrace()
		}
		
	}
	
	def getResult(tableName: String, rowKey: String): Result = {
		val get: Get = new Get(Bytes.toBytes(rowKey))
		val table = connection.getTable(TableName.valueOf(tableName))
		table.get(get)
	}
}
