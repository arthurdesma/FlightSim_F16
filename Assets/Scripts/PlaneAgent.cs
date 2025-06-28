using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// An ML-Agent that learns to fly a plane towards a series of targets within a time limit.
/// This agent uses a complex reward function to encourage efficient, stable, and safe flight.
/// It uses a Ray Perception Sensor to see and avoid terrain.
/// It can optionally be enhanced with a Genetic Algorithm to bootstrap learning.
/// </summary>
public class PlaneAgent : Agent
{
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private Rigidbody planeRigidbody;
    [SerializeField] private Transform target;

    [Header("Episode Settings")]
    [SerializeField] private float targetSpawnRadius = 500f;
    [SerializeField] private float minSpawnHeight = 50f;
    [SerializeField] private float targetReachedRadius = 40f; 
    [SerializeField] private float maxEpisodeSeconds = 300f; // The max "lifetime" of the plane in seconds

    [Header("Primary Rewards & Penalties")]
    [SerializeField] private float targetReachedReward = 25.0f;
    [SerializeField] private float crashPenalty = -15.0f;
    [SerializeField] private float timeStepPenalty = -0.005f;
    [SerializeField] private float timeOutPenalty = -5f; // Penalty for running out of time

    [Header("Guidance & Style Rewards")]
    [SerializeField] private float alignmentScale = 5.0f;
    [SerializeField] private float velocityTowardsTargetScale = 0.02f;
    [SerializeField] private float altitudeScale = 1.0f;
    [SerializeField] private float optimalSpeedScale = 0.5f;
    [SerializeField] private float stabilityScale = 0.5f;
    [SerializeField] private float progressRewardScale = 1.5f; 

    [Header("Safety & Control Penalties")]
    [SerializeField] private float stallPenaltyScale = -2.0f;
    [SerializeField] private float groundProximityPenaltyScale = -5.0f;
    [SerializeField] private float excessiveAngularVelocityPenaltyScale = -0.1f;
    
    [Header("Terrain Avoidance")]
    [SerializeField] private float terrainProximityPenaltyScale = -10f;
    [SerializeField] private float terrainCheckDistance = 250f;
    [SerializeField] private float terrainCheckSphereRadius = 15f;

    [Header("Flight Parameters")]
    [SerializeField] private float optimalSpeed = 80.0f;
    [SerializeField] private float speedTolerance = 25.0f;
    [SerializeField] private float stallAngleThreshold = 17.0f; // In degrees
    [SerializeField] private float maxAllowedAngularVelocity = 2.0f; // Radians per second

    [Header("Genetic Algorithm Enhancement")]
    [SerializeField] private bool useGeneticBootstrap = true;
    [SerializeField] private float mutationRate = 0.1f;
    [SerializeField] private float explorationDecay = 0.995f;

    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool isReady = false;
    
    // --- State tracking variables ---
    private float lastDistanceToTarget; // For progress rewards
    private float episodeTimer; // For airplane lifetime/fuel
    private bool isEpisodeActive; // Flag to ensure EndEpisode() is called only once

    // Performance tracking
    private float totalReward = 0f;

    // Genetic algorithm components
    private static List<GeneticMemory> populationMemory = new List<GeneticMemory>();
    private GeneticMemory currentGenes;
    private float explorationRate = 1.0f;

    [System.Serializable]
    private class GeneticMemory
    {
        public float fitness;
        public float[] genes;
        public float avgSpeed;
        public float avgAltitude;

        public GeneticMemory(int geneCount)
        {
            genes = new float[geneCount];
            for (int i = 0; i < geneCount; i++)
            {
                genes[i] = Random.Range(-1f, 1f);
            }
        }

        public GeneticMemory Clone()
        {
            var clone = new GeneticMemory(genes.Length);
            System.Array.Copy(genes, clone.genes, genes.Length);
            clone.fitness = fitness;
            return clone;
        }
    }

    // --- INITIALIZATION & EPISODE MANAGEMENT ---

    public override void Initialize()
    {
        InitializeAgent();
        if (useGeneticBootstrap && populationMemory.Count == 0)
        {
            for (int i = 0; i < 10; i++)
            {
                populationMemory.Add(new GeneticMemory(4));
            }
        }
        currentGenes = new GeneticMemory(4);
    }

    private void InitializeAgent()
    {
        if (isReady) return;
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        if (plane == null) plane = GetComponent<Plane>();
        if (planeRigidbody == null) planeRigidbody = GetComponent<Rigidbody>();
        isReady = true;
    }

    public override void OnEpisodeBegin()
    {
        isEpisodeActive = true; // Set the episode as active

        if (useGeneticBootstrap && totalReward != 0)
        {
            UpdateGeneticPopulation();
        }

        totalReward = 0f;
        episodeTimer = 0f; // Reset the timer
        if (!isReady) InitializeAgent();

        // Reset plane physics and position
        if (planeRigidbody != null)
        {
            planeRigidbody.velocity = Vector3.zero;
            planeRigidbody.angularVelocity = Vector3.zero;
        }
        transform.SetPositionAndRotation(initialPosition, initialRotation);

        if (plane != null)
        {
            plane.ResetPlane();
            planeRigidbody.velocity = transform.forward * 50f;
        }

        MoveTargetToRandomPosition();

        if (useGeneticBootstrap)
        {
            SelectGenesForEpisode();
        }
        
        // Initialize the distance tracker
        if(target != null)
        {
            lastDistanceToTarget = Vector3.Distance(transform.position, target.position);
        }
    }

    // --- OBSERVATIONS & ACTIONS ---

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!isReady || plane == null || planeRigidbody == null || target == null) return;

        // Total of 14 observations from this script
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.velocity)); // 3
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.angularVelocity)); // 3

        Vector3 dirToTarget = (target.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(dirToTarget)); // 3
        float currentDistance = Vector3.Distance(transform.position, target.position);
        sensor.AddObservation(currentDistance); // 1

        sensor.AddObservation(plane.Throttle); // 1
        sensor.AddObservation(plane.AngleOfAttack); // 1
        sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up)); // 1

        // Add remaining time as a normalized observation
        float remainingTimeNormalized = Mathf.Max(0, 1f - (episodeTimer / maxEpisodeSeconds));
        sensor.AddObservation(remainingTimeNormalized); // 1
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (plane == null) return;

        float pitch = actions.ContinuousActions[0];
        float yaw = actions.ContinuousActions[1];
        float roll = actions.ContinuousActions[2];
        float throttle = actions.ContinuousActions[3];

        if (useGeneticBootstrap && currentGenes != null)
        {
            float influence = Mathf.Max(0.2f, explorationRate);
            pitch = Mathf.Lerp(pitch, currentGenes.genes[0], influence);
            yaw = Mathf.Lerp(yaw, currentGenes.genes[1], influence);
            roll = Mathf.Lerp(roll, currentGenes.genes[2], influence);
            throttle = Mathf.Lerp(throttle, currentGenes.genes[3], influence);
        }

        plane.SetControlInput(new Vector3(pitch, yaw, -roll));
        plane.SetThrottleInput(throttle);

        // We calculate rewards in FixedUpdate to align with physics, but OnActionReceived is fine too.
        // For simplicity, let's stick to the original design.
        CalculateRewards();
    }
    
    /// <summary>
    /// Update the episode timer and check for timeout.
    /// </summary>
    private void FixedUpdate()
    {
        if (!isEpisodeActive) return;
        
        episodeTimer += Time.fixedDeltaTime;
        if (episodeTimer >= maxEpisodeSeconds)
        {
            AddReward(timeOutPenalty);
            TerminateEpisode(); // End the episode if we run out of time
        }
    }


    // --- REWARD CALCULATION ---

    private void CalculateRewards()
    {
        if (!isEpisodeActive || target == null || plane == null || plane.Dead) return;

        // --- Progress Reward ---
        float currentDistance = Vector3.Distance(transform.position, target.position);
        float distanceDelta = lastDistanceToTarget - currentDistance;
        if (distanceDelta > 0)
        {
            AddReward(distanceDelta * progressRewardScale);
        }
        lastDistanceToTarget = currentDistance;


        Vector3 dirToTarget = (target.position - transform.position).normalized;

        // --- 1. GUIDANCE & STYLE ---
        AddReward(Mathf.Pow((Vector3.Dot(transform.forward, dirToTarget) + 1f) / 2f, 4) * alignmentScale);
        AddReward(Mathf.Max(0, Vector3.Dot(planeRigidbody.velocity.normalized, dirToTarget)) * velocityTowardsTargetScale);
        float speedError = Mathf.Abs(plane.LocalVelocity.z - optimalSpeed);
        float speedReward = (speedError < speedTolerance) ? Mathf.Exp(-0.05f * speedError) : -(speedError / speedTolerance);
        AddReward(speedReward * optimalSpeedScale);
        float altitudeDifference = Mathf.Abs(transform.position.y - target.position.y);
        AddReward(Mathf.Exp(-0.01f * altitudeDifference) * altitudeScale);
        AddReward(((Vector3.Dot(transform.up, Vector3.up) + 1f) / 2f) * stabilityScale);

        // --- 2. SAFETY & PENALTIES ---
        if (Mathf.Abs(plane.AngleOfAttack) > stallAngleThreshold * Mathf.Deg2Rad) AddReward((Mathf.Abs(plane.AngleOfAttack) - (stallAngleThreshold * Mathf.Deg2Rad)) * stallPenaltyScale);
        if (transform.position.y < minSpawnHeight) AddReward(Mathf.Pow((minSpawnHeight - transform.position.y) / minSpawnHeight, 2) * groundProximityPenaltyScale);
        CalculateTerrainAvoidancePenalty();
        if (planeRigidbody.angularVelocity.magnitude > maxAllowedAngularVelocity) AddReward((planeRigidbody.angularVelocity.magnitude - maxAllowedAngularVelocity) * excessiveAngularVelocityPenaltyScale);

        // --- 3. PRIMARY OBJECTIVE & EFFICIENCY ---
        AddReward(timeStepPenalty);
        
        // --- Success Bubble Check ---
        if (currentDistance < targetReachedRadius)
        {
            AddReward(targetReachedReward);
            if (useGeneticBootstrap && currentGenes != null) currentGenes.fitness += targetReachedReward;
            MoveTargetToRandomPosition();
            lastDistanceToTarget = Vector3.Distance(transform.position, target.position);
        }
        
        totalReward = GetCumulativeReward();

        // --- Episode End Conditions ---
        if (plane.Dead || transform.position.y < 0)
        {
            SetReward(crashPenalty);
            TerminateEpisode();
        }
    }
    
    private void CalculateTerrainAvoidancePenalty()
    {
        if (Physics.SphereCast(transform.position, terrainCheckSphereRadius, planeRigidbody.velocity.normalized, out RaycastHit hit, terrainCheckDistance))
        {
            // Make sure the ray hit is actually terrain
            if (hit.collider.CompareTag("Terrain"))
            {
                float proximityRatio = 1f - (hit.distance / terrainCheckDistance);
                float penalty = Mathf.Pow(proximityRatio, 2) * terrainProximityPenaltyScale;
                AddReward(penalty);
            }
        }
    }
    
    private void OnCollisionEnter(Collision collision)
    {
        // Ignore collisions with the target
        if (target != null && collision.gameObject.transform == target) return;
        
        // Apply penalty based on what was hit
        if (collision.gameObject.CompareTag("Terrain")) SetReward(crashPenalty);
        else AddReward(crashPenalty * 0.5f);
        
        TerminateEpisode();
    }
    
    // --- HELPER METHOD TO SAFELY END EPISODE ---
    private void TerminateEpisode()
    {
        if (isEpisodeActive)
        {
            isEpisodeActive = false;
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // Pitch
        continuousActionsOut[0] = Input.GetKey(KeyCode.S) ? 1f : (Input.GetKey(KeyCode.W) ? -1f : 0f);
        // Yaw
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        // Roll
        continuousActionsOut[2] = Input.GetKey(KeyCode.D) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        // Throttle
        continuousActionsOut[3] = Input.GetKey(KeyCode.LeftShift) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }

    // --- CORRECTED METHOD TO PREVENT FREEZING ---
    private void MoveTargetToRandomPosition()
    {
        if (target == null) return;

        // --- FIX START: Add a sanity check to prevent an infinite loop ---
        // This validates that the spawn radius is larger than the target radius.
        // If it's not, it's impossible to find a valid point, which causes the old code to freeze.
        if (targetSpawnRadius <= targetReachedRadius)
        {
            Debug.LogError("`targetSpawnRadius` must be greater than `targetReachedRadius` to prevent an infinite loop. Check your agent's Inspector settings.");
            // Place target at a default safe position to avoid freezing the editor.
            target.position = transform.position + (Vector3.up * 100f) + (transform.forward * 100f);
            return;
        }
        // --- FIX END ---

        Vector3 randomPosition;
        int maxAttempts = 100; // --- FIX: Add a maximum attempt counter ---

        for (int i = 0; i < maxAttempts; i++)
        {
            // Get a random point within a sphere around the agent's CURRENT position.
            randomPosition = transform.position + (Random.insideUnitSphere * targetSpawnRadius);
            
            // Ensure the target is not below the minimum spawn height.
            randomPosition.y = Mathf.Max(randomPosition.y, minSpawnHeight);

            // If the randomly chosen point is a safe distance away, we found our spot.
            if (Vector3.Distance(transform.position, randomPosition) > targetReachedRadius)
            {
                target.position = randomPosition;
                return;
            }
        }
        
        // --- FIX: If the loop finishes without finding a point (highly unlikely with correct settings) ---
        Debug.LogWarning("Could not find a valid random position for the target after " + maxAttempts + " attempts. Placing it at a default location.");
        target.position = transform.position + (transform.forward * (targetSpawnRadius * 0.5f)) + (Vector3.up * minSpawnHeight);
    }

    private void SelectGenesForEpisode()
    {
        if (Random.value < explorationRate || populationMemory.All(g => g.fitness == 0))
        {
            currentGenes = new GeneticMemory(4);
        }
        else
        {
            var bestGenes = populationMemory.OrderByDescending(g => g.fitness).First();
            currentGenes = bestGenes.Clone();
            for (int i = 0; i < currentGenes.genes.Length; i++)
            {
                if (Random.value < mutationRate)
                {
                    currentGenes.genes[i] = Mathf.Clamp(currentGenes.genes[i] + Random.Range(-0.3f, 0.3f), -1f, 1f);
                }
            }
        }
        explorationRate = Mathf.Max(0.1f, explorationDecay * explorationRate);
    }

    private void UpdateGeneticPopulation()
    {
        currentGenes.fitness = GetCumulativeReward();
        currentGenes.avgSpeed = plane.LocalVelocity.magnitude;
        currentGenes.avgAltitude = transform.position.y;

        var worstPerformer = populationMemory.OrderBy(g => g.fitness).First();

        if (currentGenes.fitness > worstPerformer.fitness)
        {
            int index = populationMemory.IndexOf(worstPerformer);
            populationMemory[index] = currentGenes.Clone();
        }
    }
}


// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --force

// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --resume