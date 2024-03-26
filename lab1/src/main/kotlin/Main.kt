import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.time.measureTime
import krangl.DataFrame
import krangl.max
import krangl.min
import krangl.readCSV
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.feature.layout
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.kandy.letsplot.settings.LineType
import org.jetbrains.kotlinx.kandy.letsplot.x
import org.jetbrains.kotlinx.kandy.letsplot.y
import org.jetbrains.kotlinx.kandy.util.color.Color
import smile.math.distance.Distance
import smile.math.distance.EuclideanDistance
import smile.math.distance.ManhattanDistance

const val INF = 1e5
const val EPS = 1e-13
const val PERCENTAGE = 0.95
const val MAX_WINDOW_PARAM = 15

enum class CoreType {
    UNIFORM,
    TRIANGULAR,
    EPANECHNIKOV,
    GAUSSIAN,
}

enum class MetricType {
    COSINE,
    EUCLIDEAN,
    MANHATTAN,
}

sealed class WindowType

data class FixedWindow(
    val h: Int,
) : WindowType()

data class VariableWindow(
    val k: Int,
) : WindowType()

data class SolutionSetup(
    val coreType: CoreType,
    val metricType: MetricType,
    val windowType: WindowType,
)

fun metric(type: MetricType): (List<Double>, List<Double>) -> Double {
    return when (type) {
        MetricType.COSINE -> { a, b ->
            val aSum = a.sumOf { it * it }
            val bSum = b.sumOf { it * it }
            val s = a.mapIndexed { index, d -> d * b[index] }.sum()
            s / (aSum * bSum)
        }

        MetricType.EUCLIDEAN -> { a, b ->
            sqrt((a zip b).sumOf { (it.first - it.second) * (it.first - it.second) })
        }

        MetricType.MANHATTAN -> { a, b ->
            (a zip b).sumOf { abs(it.first - it.second) }
        }
    }
}

fun core(type: CoreType): (u: Double) -> Double {
    return when (type) {
        CoreType.UNIFORM -> { u -> if (u < 1.0) 0.5 else 0.0 }
        CoreType.TRIANGULAR -> { u -> if (u < 1.0) 1.0 - u else 0.0 }
        CoreType.EPANECHNIKOV -> { u -> if (u < 1.0) 0.75 * (1 - u * u) else 0.0 }
        CoreType.GAUSSIAN -> { u -> 1 / sqrt(2 * PI) * exp(-u * u / 2) }
    }
}

/* suppose k in [1..m], where m = x.size*/
fun knn(
    coreType: CoreType,
    metricType: MetricType,
    windowType: WindowType,
    u: List<Double>,
    x: List<List<Double>>,
    y: List<Int>,
    w: List<Double>,
): Int {
    val coreMethod = core(coreType)
    val metricMethod = metric(metricType)

    val sortedNeighbours = x
        .mapIndexed { index, doubles -> Pair(doubles, y[index]) }
        .sortedBy { metricMethod(u, it.first) }

    val m = x.size
    var mx = -INF
    var answer = -1

    val sums: HashMap<Long, Double> = HashMap()

    for (i in (0..<m)) {
        sums[sortedNeighbours[i].second.toLong()] =
            sums.getOrDefault(sortedNeighbours[i].second.toLong(), 0.0) + when (windowType) {
                is FixedWindow -> {
                    val a = metricMethod(u, sortedNeighbours[i].first)
                    val b = windowType.h
                    w[i] * coreMethod(a / b)
                }

                is VariableWindow -> {
                    val a = metricMethod(u, sortedNeighbours[i].first)
                    val b = metricMethod(
                        u,
                        sortedNeighbours[windowType.k - 1].first
                    )
                    w[i] * coreMethod(a / b)
                }
            }
        if (sums[sortedNeighbours[i].second.toLong()]!! > mx) {
            mx = sums[sortedNeighbours[i].second.toLong()]!!
            answer = sortedNeighbours[i].second
        }
    }

    return answer
}


fun buildConfusionMatrix(yReal: List<Int>, yPred: List<Int>, classCount: Int): Array<Array<Int>> {
    val m = yReal.size
    val confusionMatrix = Array(classCount + 1) { _ -> Array(classCount + 1) { _ -> 0 } }

    for (i in (0..<m)) {
        confusionMatrix[yPred[i]][yReal[i]] += 1
    }

    return confusionMatrix
}

fun isZero(a: Double): Boolean {
    return abs(a) < EPS
}

fun precision(confusionMatrix: Array<Array<Int>>): Double {
    val m = confusionMatrix.size
    return (0..<m).map { c ->
        val s = (0..<m).sumOf { i -> confusionMatrix[c][i] }.toDouble()
        if (isZero(s)) -INF
        else if (isZero(confusionMatrix[c][c].toDouble())) 0.0
        else confusionMatrix[c][c] / s
    }.filter { it != -INF }.average()
}

fun recall(confusionMatrix: Array<Array<Int>>): Double {
    val m = confusionMatrix.size
    return (0..<m).map { c ->
        val s = (0..<m).sumOf { i -> confusionMatrix[c][i] }.toDouble()
        if (isZero(s)) -INF
        else if (isZero(confusionMatrix[c][c].toDouble())) 0.0
        else confusionMatrix[c][c] / s
    }.filter { it != -INF }.average()
}

fun calcFScore(confusionMatrix: Array<Array<Int>>): Double {
    val precision = precision(confusionMatrix)

    val recall = recall(confusionMatrix)

    return if (isZero(precision + recall)) 0.0
    else 2 * precision * recall / (precision + recall)
}

fun letsPlot(l: List<Pair<Int, Double>>, fileName: String) {
    val first = l.map { it.first }
    val second = l.map { it.second }

    val kData = mapOf(
        "k" to first,
        "f-score" to second,
    )

    val simplePlot = plot(kData) {
        x("k")
        y("f-score")
        line {
            width = 3.0 // Set line width
            color = Color.hex("#6e5596") // Define line color
            type = LineType.SOLID // Specify the line type
        }

        layout { // Set plot layout
            title = "f-score(k)" // Add title
            // Add caption
            caption = "graph of f-score(k)"
            size = 700 to 450 // Plot dimension settings
        }
    }
    simplePlot.save(fileName)
}

fun main() {
    val df = DataFrame.readCSV("winequality-red.csv")
    val classes: List<Int> = df.cols.last().values().map { it.toString().toInt() }.toList()

    /*
    * [3..8]
    * println(classes.minOrNull())
    * println(classes.maxOrNull())
    * */

    /* 249 */
    val sizeDf = df.nrow
    println(sizeDf)

    val dfObjects = df.remove("quality")
    val minMax: List<Pair<Double, Double>> = dfObjects.cols.map { Pair(it.min()!!, it.max()!!) }
    val objects: Array<List<Double>> = Array(sizeDf) {
        dfObjects.row(it).values.mapIndexed { i, el ->
            (el.toString().toDouble() - minMax[i].first) / (minMax[i].second - minMax[i].first)
        }
    }

    /* split on train and test sets */
    val xTrain = objects.dropLast(((1 - PERCENTAGE) * sizeDf).toInt())
    val xTest = objects.drop(xTrain.size)

    val yTrain = classes.dropLast(((1 - PERCENTAGE) * sizeDf).toInt())
    val yTest = classes.drop(yTrain.size)

    var mx = 0.0
    var bestSolutionSetup: SolutionSetup? = null
    val timeOnTraining = measureTime {
        for (coreType in CoreType.entries) {
            for (metricType in MetricType.entries) {
                for (window in (0..1)) {
                    for (t in (1..MAX_WINDOW_PARAM)) {
                        val windowType = if (window == 0) {
                            FixedWindow(t)
                        } else {
                            VariableWindow(t)
                        }
                        val yPred = (xTrain.indices).map {
                            knn(
                                coreType = coreType,
                                metricType = metricType,
                                windowType = windowType,
                                u = xTrain[it],
                                x = xTrain.filterIndexed { index, _ -> index != it },
                                y = yTrain,
                                w = (yTrain.indices).map { 1.0 }
                            )
                        }
                        val `f-score` = calcFScore(buildConfusionMatrix(yTrain, yPred, classes.max()))
                        if (`f-score` >= mx) {
                            mx = `f-score`
                            bestSolutionSetup = SolutionSetup(
                                coreType = coreType,
                                metricType = metricType,
                                windowType = windowType,
                            )
                        }
                    }
                }
            }
        }
    }


    println(mx)
    println(bestSolutionSetup)

    println("Time spent on training=$timeOnTraining")

    val yPredMine = xTest.indices.map {
        knn(
            coreType = bestSolutionSetup!!.coreType,
            metricType = bestSolutionSetup!!.metricType,
            windowType = bestSolutionSetup!!.windowType,
            u = xTest[it],
            x = xTest.filterIndexed { index, _ -> index != it },
            y = yTest,
            w = (yTest.indices).map { 1.0 },
        )
    }

    /* LOWESS */
    val gaussian = core(CoreType.GAUSSIAN)
    var yPredLowess = yPredMine
    for (iters in (0..5)) {
        val w = yPredLowess.mapIndexed { index, i ->
            val value = if (yTest[index] != i) 0.0 else 1.0
            gaussian(value)
        }
        val yPredIter = xTest.indices.map {
            knn(
                coreType = bestSolutionSetup!!.coreType,
                metricType = bestSolutionSetup!!.metricType,
                windowType = bestSolutionSetup!!.windowType,
                u = xTest[it],
                x = xTest.filterIndexed { index, _ -> index != it },
                y = yTest,
                w = w,
            )
        }
        yPredLowess = yPredIter
    }

    val metrics = listOf(EuclideanDistance(), ManhattanDistance())
    var fMaxLib = -1.0

    var bestMetric: Distance<DoubleArray>? = null
    var bestK = -1
    for (distType in metrics) {
        for (t in (1..MAX_WINDOW_PARAM)) {
            val knnLib = smile.classification.knn(
                x = xTrain.map { it.toDoubleArray() }.toTypedArray(),
                y = yTrain.toIntArray(),
                k = t,
                distance = distType,
            )
            val yPred = (xTrain.indices).map {
                knnLib.predict(xTrain[it].toDoubleArray())
            }
            val `f-score` = calcFScore(buildConfusionMatrix(yTrain, yPred, classes.max()))
            if (`f-score` >= fMaxLib) {
                fMaxLib = `f-score`
                bestMetric = distType
                bestK = t
            }
        }
    }

    val knnLib = smile.classification.knn(
        x = xTrain.map { it.toDoubleArray() }.toTypedArray(),
        y = yTrain.toIntArray(),
        k = bestK,
        distance = bestMetric!!,
    )
    val yPredLib = (xTest.indices).map {
        knnLib.predict(xTest[it].toDoubleArray())
    }

    val confusionMatrixMine = buildConfusionMatrix(yTest, yPredMine, classes.max())
    val confusionMatrixLowess = buildConfusionMatrix(yTest, yPredLowess, classes.max())
    val confusionMatrixLib = buildConfusionMatrix(yTest, yPredLib, classes.max())
    val fScoreMine = calcFScore(confusionMatrixMine)
    val fScoreLowwss = calcFScore(confusionMatrixLowess)
    val fScoreLib = calcFScore(confusionMatrixLib)
    println("f-score-mine=$fScoreMine")
    println("precision mine=${precision(confusionMatrixMine)}")
    println("f-score-lowess=$fScoreLowwss")
    println("precision lowess=${precision(confusionMatrixLowess)}")
    println("f-score-lib=$fScoreLib")
    println("precision lib=${precision(confusionMatrixLib)}")
    /* our is better because they don't have cores! */


    val pointsOnPlotTest = (1..MAX_WINDOW_PARAM).map {
        val yPredCur = xTest.indices.map { ind ->
            knn(
                coreType = bestSolutionSetup!!.coreType,
                metricType = bestSolutionSetup!!.metricType,
                windowType = FixedWindow(it),
                u = xTest[ind],
                x = xTest.filterIndexed { index, _ -> index != ind },
                y = yTest,
                w = (yTest.indices).map { 1.0 },
            )
        }
        Pair(it, calcFScore(buildConfusionMatrix(yTest, yPredCur, classes.max())))
    }
    letsPlot(pointsOnPlotTest, "testSetting.png")

    val pointsOnPlotTrain = (1..MAX_WINDOW_PARAM).map {
        val yPredCur = xTrain.indices.map { ind ->
            knn(
                coreType = bestSolutionSetup!!.coreType,
                metricType = bestSolutionSetup!!.metricType,
                windowType = FixedWindow(it),
                u = xTrain[ind],
                x = xTrain.filterIndexed { index, _ -> index != ind },
                y = yTrain,
                w = (yTrain.indices).map { 1.0 },
            )
        }
        Pair(it, calcFScore(buildConfusionMatrix(yTrain, yPredCur, classes.max())))
    }
    letsPlot(pointsOnPlotTrain, "trainSetting.png")

    val pointsOnPlotTestLib = (1..MAX_WINDOW_PARAM).map {
        val knnLibPlot = smile.classification.knn(
            x = xTrain.map { x ->  x.toDoubleArray() }.toTypedArray(),
            y = yTrain.toIntArray(),
            k = it,
            distance = bestMetric,
        )
        val yPredLibPlot = (xTest.indices).map { ind ->
            knnLibPlot.predict(xTest[ind].toDoubleArray())
        }
        Pair(it, calcFScore(buildConfusionMatrix(yTest, yPredLibPlot, classes.max())))
    }
    letsPlot(pointsOnPlotTestLib, "testSettingLib.png")

    val pointsOnPlotTrainLib = (1..MAX_WINDOW_PARAM).map {
        val knnLibPlot = smile.classification.knn(
            x = xTrain.map { x ->  x.toDoubleArray() }.toTypedArray(),
            y = yTrain.toIntArray(),
            k = it,
            distance = bestMetric,
        )
        val yPredLibPlot = (xTrain.indices).map { ind ->
            knnLibPlot.predict(xTrain[ind].toDoubleArray())
        }
        Pair(it, calcFScore(buildConfusionMatrix(yTrain, yPredLibPlot, classes.max())))
    }
    letsPlot(pointsOnPlotTrainLib, "trainSettingLib.png")
}