package com.fzr.util

import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.unsafe.hash.Murmur3_x86_32.hashUnsafeBytes
import org.apache.spark.unsafe.types.UTF8String

class DataUtil {
	
}
object DataUtil{
	/**
	 * 在一个double数组中二分搜索指定值
	 * @param valSet
	 * @param value
	 * @return
	 */
	def binarySearch(valSet: List[Double], value: Double):Int={
		val sz = valSet.size
		if (value < valSet.head) 0     // 0
		else if(value > valSet.last) valSet.size-1
		// 找到第一个大于该value的index
		else {
			var (low, high) = (0, sz - 1)
			var mid: Int = 0
			while (low < high) {
				mid = low + (high - low) / 2
				if (value < valSet(mid)) high = mid
				else low = mid+1
			}
			low    // 0~sz-1
		}
	}
	
	/**
	 * 根据路径配置加载配置文件
	 * @param filePath
	 */
	def loadTextFile(filePath: String) = {
		val stream : InputStream = getClass.getResourceAsStream(filePath)
		val lines = scala.io.Source.fromInputStream( stream ).getLines.toArray.map(_.trim)
		lines
	}
	
	/**
	 * 根据路径读取文件内容字符串
	 * @param filePath
	 * @return
	 */
	def loadStringFromFile(filePath: String)={
		val stream : InputStream = getClass.getResourceAsStream(filePath)
		val lines = scala.io.Source.fromInputStream( stream ).mkString
		lines
	}
	/**
	 * 对输入字符串做hash操作
	 * @param x
	 * @param num
	 * @return
	 */
	def makeHash(x: String, num: Int): Int = {
		def nonNegativeMod(x: Int, mod: Int): Int = {
			val rawMod = x % mod
			rawMod + (if (rawMod < 0) mod else 0)
		}
		
		if (x == null || x.equals("") || x.equals("0")) return 0
		val s = UTF8String.fromString(x)
		nonNegativeMod(hashUnsafeBytes(s.getBaseObject, s.getBaseOffset, s.numBytes(), 42), num)
	}
	
	/**
	 * 将各种数字的字符串转成数字
	 * @param x
	 * @return
	 */
	def getValueFromString(x: String): Int={
		if(x == null || x.equals("") || x.equals("None") || x.equals("NULL")) return 0
		x.toInt
	}
	/**
	 * 获取指定日期的N天以前的日期, N<0则为N天以后的日期
	 * @param dt
	 * @param dayNum
	 * @return
	 */
	def getPreNDay(dt: String, dayNum: Int, formatStr: String = "yyyyMMdd"): String = {
		val format = new SimpleDateFormat(formatStr) //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		calendar.add(Calendar.DATE, -dayNum)  // N天前的日期
		classDate = calendar.getTime //获取加减以后的Date类型日期
		val prenDay = format.format(classDate)
		//		println("prenDay=", prenDay)
		prenDay
	}
	
	def dateBefore(dt1: String, dt2: String, formatStr: String="yyyyMMdd"): Boolean ={
		val format = new SimpleDateFormat(formatStr) //定义日期格式化的格式
		var classDate1 = format.parse(dt1) //把字符串转化成指定格式的日期
		var classDate2 = format.parse(dt2)
		classDate1.before(classDate2)
	}
	/**
	 * 获取今天的日期
	 * @return
	 */
	def getTodayDt(): String ={
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		val today = format.format(calendar.getTime)
		today
	}
	/**
	 * 获取指定日期的上月日期
	 * @param dt
	 */
	def getCurMonth(dt: String): String ={
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		classDate = calendar.getTime //获取加减以后的Date类型日期
		val monthFormat = new SimpleDateFormat("yyyyMM")
		val curMonth = monthFormat.format(classDate)
		curMonth
	}
	
	/**
	 * 获取指定日期的上月日期
	 * @param dt
	 */
	def getLastMonthDate(dt: String): String ={
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		calendar.add(Calendar.MONTH, -1)  // N天前的日期
		classDate = calendar.getTime //获取加减以后的Date类型日期
		val monthFormat = new SimpleDateFormat("yyyyMM")
		val lastMonth = monthFormat.format(classDate)
		lastMonth
	}
	/**
	 * 获取指定日期的上月本日
	 * @param dt
	 */
	def getLastMonthToday(dt: String): String ={
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		calendar.add(Calendar.MONTH, -1)  // N天前的日期
		classDate = calendar.getTime //获取加减以后的Date类型日期
		val lastMonthToday = format.format(classDate)
		lastMonthToday
	}
	
	/**
	 * 获取上个月的最后一天
	 * @param dt
	 * @return
	 */
	def getLastMonthLastDay(dt:String): String={
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		calendar.add(Calendar.MONTH, -1)  // N天前的日期
		calendar.set(Calendar.DAY_OF_MONTH, calendar.getActualMaximum(Calendar.DATE))
		classDate = calendar.getTime //获取加减以后的Date类型日期
		val lastMonthLastDay = format.format(classDate)
		lastMonthLastDay
	}
	/**
	 * 获取某个日期当前月份的剩余天数
	 * @param dt
	 * @return
	 */
	def getDateMonthRemainDays(dt: String): Int = {
		val format = new SimpleDateFormat("yyyyMMdd") //定义日期格式化的格式
		var classDate = format.parse(dt) //把字符串转化成指定格式的日期
		val calendar = Calendar.getInstance //使用Calendar日历类对日期进行加减
		calendar.setTime(classDate)
		val curMonthAllDays: Int = calendar.getActualMaximum(Calendar.DAY_OF_MONTH) // 本月天数
		val curDay: Int = calendar.get(Calendar.DAY_OF_MONTH) // 当前天数
		
		curMonthAllDays - curDay
	}
	/**
	 * 解析json
	 *
	 * @param json
	 * @return
	 */
	def regJson(json: Option[Any]) = json match {
		case Some(map: Map[String, Any]) => map
	}
	
	/**
	 * 解析List
	 *
	 * @param tmp
	 * @return
	 */
	def regList(tmp: Option[Any]) = tmp match {
		case Some(list: List[String]) => list
	}
	
	/**
	 * 解析String
	 *
	 * @param tmp
	 * @return
	 */
	def regString(tmp: Option[Any]) = tmp match {
		case Some(str: String) => str
	}
	def boolean2Int(tmp: Boolean) :Int = {
		if(tmp) 0 else 1
	}
	/**
	 * 输入：日期
	 * 输出：日期对应星期几
	 * @param dt
	 * @return
	 */
	def getWeek(dt: String): Int = {
		val weekDays = Array(7, 1, 2, 3, 4, 5, 6)
		val date = new SimpleDateFormat("yyyyMMdd").parse(dt)
		val cal = Calendar.getInstance()
		cal.setTime(date)
		var week = cal.get(Calendar.DAY_OF_WEEK) - 1
		if(week < 0) {
			week = 0
		}
		weekDays(week)
	}
}
