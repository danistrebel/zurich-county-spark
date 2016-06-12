package io.github.danistrebel.zurichcountyspark

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors

case class DemographicsEntry(
                            bfsId: Int,
                            name: String,
                            year: Int,
                            quota: Double
                            )

case class DemographicsSummary(bfsId: Int, name: String, latestDemographicQuota: Double, demographicTrend: Double)

object DemographicsClustering extends App {

  val conf = new SparkConf().setAppName("Zurich Demographics").setMaster("local[4]")

  val sc = new SparkContext(conf)

  val data = sc.textFile(getClass.getResource("/age20-39.csv").getPath)

  val entries = data.map(line => {
    val splits = line.split(";")
    DemographicsEntry(splits(0).toInt, splits(1), splits(7).toInt, splits(8).toDouble)
  })

  val demographicChange = entries.groupBy(_.name).values.map(r => {
    val annualQuotas = r.toList.sortBy(_.year)
    val latest = annualQuotas.last
    val oldest = annualQuotas.head
    DemographicsSummary(r.head.bfsId, r.head.name, latest.quota, latest.quota-oldest.quota)
  })

  val vectorPairs = demographicChange.map(change => {
    (change, Vectors.dense(change.latestDemographicQuota, change.demographicTrend))
  })

  val trainData = vectorPairs.map(_._2)

  val numClusters = 3
  val numIterations = 20
  val clusters = KMeans.train(trainData, numClusters, numIterations)

  val clusterResults = vectorPairs.map({
    case (demographic, vector) => (demographic, clusters.predict(vector))
  })

  clusterResults.collect.groupBy(_._2).foreach({ case (clusterId, members) => {
    val prettyMembers = members.map(_._1).sortBy(_.name).map(x => {
      s"${clusterId},${x.name},${x.latestDemographicQuota},${x.demographicTrend}"
    })
    println(prettyMembers.mkString("\n"))
  }})
}
