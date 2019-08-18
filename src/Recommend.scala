
import java.io.File
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import scala.collection.immutable.Map

object Recommend {
  //功能集成函数
  def Recommend(
    model:      MatrixFactorizationModel,
    movieTitle: Map[Int, String]) = {
    var choose = ""
    while (choose != "3") {
      print("请选择要推荐的类型  1.针对用户推荐电影  2.针对电影推荐给感兴趣的用户  3.离开？")
      choose = readLine()
      if (choose == "1") {
        print("请输入用户id?")
        var inputUserID = readLine() //读取用户ID
        RecommendMovies(model, movieTitle, inputUserID.toInt) //针对此用户推荐电影
      } else if (choose == "2") {
        print("请输入电影id?")
        val inputMovieID = readLine() //读取电影ID
        RecommendUsers(model, movieTitle, inputMovieID.toInt) //针对此电影推荐用户
      }
    }
  }

  //设置不显示log信息
  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }

  //数据准备local
  def PrepareData_local(): (
      RDD[Rating], 
      Map[Int, String]) = {
    val DataDir = "file:/home/test/workspace/Recommend/data/"
    //-----------------1.创建用户评分数据---------------------
    val sc = new SparkContext(new SparkConf().setAppName("Recommend").setMaster("local[4]"))
    printf("开始读取用户评分数据...")
    val rawUserData = sc.textFile(new File(DataDir, "u.data").toString)
    //读取rawUserData的前3个字段
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    //准备ALS训练数据
    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }

    println("共计" + ratingsRDD.count().toString() + "条ratings")

    //-----------------2.创建电影ID与名称对应表----------------
    printf("开始读取电影数据中...")
    val itemRDD = sc.textFile(new File(DataDir, "u.item").toString)
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collect().toMap

    //---------------3.显示数据记录数-------------------------
    val numRatings = ratingsRDD.count()
    val numUser = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共计：ratings: " + numRatings + " User :" + numUser + " Movie " + numMovies)
    //返回模型训练数据和电影ID名称对应表
    return (ratingsRDD, movieTitle)
  }

  //数据准备hdfs
//  def PrepareData_hdfs():(
//      RDD[Rating],
//      Map[Int,String]) = {
//      
//    
//  }
  
  //创建、训练模型
  def TrainModel(
    ratingsRDD: RDD[Rating]):MatrixFactorizationModel = {
    println("开始训练模型...")
    val model = ALS.train(ratingsRDD, 10, 10, 0.01)
    return model
  }

  //针对此用户推荐电影
  def RecommendMovies(
    model:       MatrixFactorizationModel,
    movieTitle:  Map[Int, String],
    inputUserID: Int) = {
    val RecommendMovie = model.recommendProducts(inputUserID, 10)
    var i = 1
    println("针对用户 id " + inputUserID + " 推荐下列电影")
    RecommendMovie.foreach {
      r =>
        println(i.toString() + "." + movieTitle(r.product) + "评分：" + r.rating.toString())
        i += 1
    }
  }

  //针对电影推荐用户
  def RecommendUsers(
    model:        MatrixFactorizationModel,
    movieTitle:   Map[Int, String],
    inputMovieID: Int) = {
    val RecommendUser = model.recommendUsers(inputMovieID, 10)
    var i = 1
    println("针对电影id" + inputMovieID + " 电影名：" + movieTitle)
    RecommendUser.foreach(r => println(i.toString() + "用户id:" + r.user + " 评分：" + r.rating))
    i = i + 1
  }

  def main(args: Array[String]): Unit = {
    SetLogger() //设置不显示log信息
    val (ratingsRDD, movieTitle) = PrepareData_local() //准备数据阶段,数据来源local
    val model = TrainModel(ratingsRDD) //训练模型
    Recommend(model, movieTitle) //选择推荐方式
  }

}