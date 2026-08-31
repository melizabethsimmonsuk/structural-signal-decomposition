import scala.math.{abs, pow, sqrt}

final case class Observation(
    index: Int,
    value: Double,
    weight: Double = 1.0
)

final case class StructuralComponent(
    label: String,
    magnitude: Double,
    persistence: Double,
    deviation: Double
)

object StructuralSignalDecomposition {

  def weightedMean(
      observations: Vector[Observation]
  ): Double = {

    val weightedTotal =
      observations.map(o => o.value * o.weight).sum

    val totalWeight =
      observations.map(_.weight).sum

    if (totalWeight == 0.0) 0.0
    else weightedTotal / totalWeight
  }

  def variance(
      observations: Vector[Observation]
  ): Double = {

    if (observations.isEmpty) 0.0
    else {
      val mean = weightedMean(observations)

      observations
        .map(o => pow(o.value - mean, 2))
        .sum / observations.size
    }
  }

  def volatility(
      observations: Vector[Observation]
  ): Double =
    sqrt(variance(observations))

  def persistence(
      observations: Vector[Observation]
  ): Double = {

    if (observations.size < 2) 0.0
    else {
      val adjacentPairs =
        observations.sliding(2).toVector

      val directionalAgreement =
        adjacentPairs.count {
          case Vector(a, b) =>
            math.signum(a.value) == math.signum(b.value)

          case _ =>
            false
        }

      directionalAgreement.toDouble /
        adjacentPairs.size
    }
  }

  def normalize(
      observations: Vector[Observation]
  ): Vector[Observation] = {

    val mean = weightedMean(observations)
    val sigma = volatility(observations)

    observations.map { observation =>
      val normalized =
        if (sigma == 0.0) 0.0
        else (observation.value - mean) / sigma

      observation.copy(value = normalized)
    }
  }

  def extractComponent(
      label: String,
      observations: Vector[Observation]
  ): StructuralComponent = {

    val mean = weightedMean(observations)
    val sigma = volatility(observations)
    val stable = persistence(observations)

    StructuralComponent(
      label = label,
      magnitude = abs(mean),
      persistence = stable,
      deviation = sigma
    )
  }

  def rankComponents(
      components: Vector[StructuralComponent]
  ): Vector[StructuralComponent] =
    components.sortBy { component =>
      -(
        component.magnitude *
        component.persistence /
        (1.0 + component.deviation)
      )
    }

  def main(args: Array[String]): Unit = {

    val observations = Vector(
      Observation(1, 0.42),
      Observation(2, 0.57),
      Observation(3, 0.61),
      Observation(4, 0.55),
      Observation(5, 0.73),
      Observation(6, 0.81),
      Observation(7, 0.76),
      Observation(8, 0.92)
    )

    val normalized =
      normalize(observations)

    val baseline =
      extractComponent(
        "baseline-structure",
        observations
      )

    val normalizedStructure =
      extractComponent(
        "normalized-residual",
        normalized
      )

    val ranked =
      rankComponents(
        Vector(
          baseline,
          normalizedStructure
        )
      )

    println("Structural Signal Decomposition")
    println("-------------------------------")

    ranked.foreach { component =>
      println(
        f"${component.label}%-24s " +
        f"magnitude=${component.magnitude}%.4f " +
        f"persistence=${component.persistence}%.4f " +
        f"deviation=${component.deviation}%.4f"
      )
    }
  }
}
