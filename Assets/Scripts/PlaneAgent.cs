using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Linq;

public class PlaneAgent : Agent
{
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private Rigidbody planeRigidbody;

    [Header("Training")]
    [SerializeField] private Transform target;
    [SerializeField] private float targetSpawnRadius = 500f;
    [SerializeField] private float minSpawnHeight = 50f;

    [Header("Reward Settings")]
    [SerializeField] private float targetReachedReward = 25.0f;
    [SerializeField] private float crashPenalty = -15.0f;
    [SerializeField] private float velocityRewardScale = 0.02f;
    [SerializeField] private float alignmentRewardScale = 5.0f;
    [SerializeField] private float altitudeRewardScale = 1.0f;
    [SerializeField] private float stabilityRewardScale = 0.5f;
    [SerializeField] private float speedRewardScale = 0.5f;
    [SerializeField] private float stallPenaltyScale = -2.0f;
    [SerializeField] private float groundProximityPenaltyScale = -5.0f;
    [SerializeField] private float timeStepPenalty = -0.005f;
    [SerializeField] private float optimalSpeed = 80.0f;
    [SerializeField] private float speedTolerance = 25.0f;


    [Header("Genetic Algorithm Enhancement")]
    [SerializeField] private bool useGeneticBootstrap = true;
    [SerializeField] private float mutationRate = 0.1f;
    [SerializeField] private float explorationDecay = 0.995f;
    
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool isReady = false;
    
    // Performance tracking
    private float episodeStartTime;
    private float totalReward = 0f;
    private int successfulTargets = 0;
    
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

    // --- Initialization Methods ---

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
        if (useGeneticBootstrap && totalReward != 0)
        {
            UpdateGeneticPopulation();
        }
        
        episodeStartTime = Time.time;
        totalReward = 0f;
        successfulTargets = 0;
        
        if (!isReady) InitializeAgent();
        
        if (planeRigidbody != null)
        {
            planeRigidbody.velocity = Vector3.zero;
            planeRigidbody.angularVelocity = Vector3.zero;
        }
        
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        
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
    }

    // --- Action & Observation ---

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!isReady) InitializeAgent();
        
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.velocity));
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.angularVelocity));
        
        Vector3 dirToTarget = (target.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(dirToTarget));
        
        sensor.AddObservation(Vector3.Distance(transform.position, target.position));
        sensor.AddObservation(plane.Throttle);
        sensor.AddObservation(plane.AngleOfAttack);
        sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up));
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
        
        CalculateRewards();
    }

    // --- NEW REWARD FUNCTION ---

    /// <summary>
    /// Calculates and applies rewards and penalties based on the plane's current state and actions.
    /// This function encourages efficient, stable flight towards the target.
    /// </summary>
    private void CalculateRewards()
    {
        if (target == null || plane == null || plane.Dead) return;
        
        // --- Pre-calculate necessary values ---
        Vector3 dirToTarget = (target.position - transform.position).normalized;
        Vector3 currentVelocity = planeRigidbody.velocity;
        
        // --- 1. Velocity Reward: Encourage flying efficiently towards the target ---
        // Rewards the component of velocity that is directed towards the target.
        float velocityTowardsTarget = Vector3.Dot(currentVelocity.normalized, dirToTarget);
        AddReward(Mathf.Max(0, velocityTowardsTarget) * velocityRewardScale);

        // --- 2. Alignment Reward: Strongly incentivize pointing the nose at the target ---
        float alignment = Vector3.Dot(transform.forward, dirToTarget);
        // Use a power function to make the reward much stronger for perfect alignment.
        // Maps alignment from [-1, 1] to a reward shaping curve in [0, 1].
        float alignmentBonus = Mathf.Pow((alignment + 1f) / 2f, 4); 
        AddReward(alignmentBonus * alignmentRewardScale);

        // --- 3. Energy Management Rewards: Speed and Altitude Control ---
        // Reward for maintaining an altitude close to the target's altitude.
        float altitudeDifference = Mathf.Abs(transform.position.y - target.position.y);
        float altitudeReward = Mathf.Exp(-0.01f * altitudeDifference); // Exponential decay reward
        AddReward(altitudeReward * altitudeRewardScale);
        
        // Reward for maintaining an optimal speed.
        float currentForwardSpeed = plane.LocalVelocity.z;
        float speedError = Mathf.Abs(currentForwardSpeed - optimalSpeed);
        // Reward is high when near optimal, but quickly becomes a penalty if outside the tolerance range.
        float speedReward = (speedError < speedTolerance) 
            ? Mathf.Exp(-0.05f * speedError) 
            : -(speedError / speedTolerance);
        AddReward(speedReward * speedRewardScale);
        
        // --- 4. Flight Stability Reward ---
        // Reward for flying upright. Gentle enough to allow for banking turns.
        float uprightness = Vector3.Dot(transform.up, Vector3.up); // -1 (upside down) to 1 (upright)
        AddReward(((uprightness + 1f) / 2f) * stabilityRewardScale);

        // --- 5. Penalties for Dangerous States ---
        // Penalty for high angle of attack (approaching a stall). Gets worse with severity.
        if (Mathf.Abs(plane.AngleOfAttack) > 0.3f) // ~17 degrees
        {
            float stallSeverity = (Mathf.Abs(plane.AngleOfAttack) - 0.3f);
            AddReward(stallSeverity * stallPenaltyScale); // scale is negative
        }
        
        // Penalty for flying too close to the ground. Gets exponentially worse.
        if (transform.position.y < minSpawnHeight)
        {
            float groundProximity = (minSpawnHeight - transform.position.y) / minSpawnHeight; // 0 to 1
            AddReward(Mathf.Pow(groundProximity, 2) * groundProximityPenaltyScale); // scale is negative
        }

        // --- 6. Constant Time Penalty ---
        // Encourages the agent to complete the task as quickly as possible.
        AddReward(timeStepPenalty);
        
        // --- Track total reward and handle episode end conditions ---
        totalReward = GetCumulativeReward();
        
        if (plane.Dead || transform.position.y < 0)
        {
            SetReward(crashPenalty);
            EndEpisode();
        }
    }

    // --- Triggers and Collisions ---

    private void OnTriggerEnter(Collider other)
    {
        if (target != null && other.transform == target)
        {
            successfulTargets++;
            AddReward(targetReachedReward); // Use the tunable reward
            MoveTargetToRandomPosition();
            
            if (useGeneticBootstrap && currentGenes != null)
            {
                currentGenes.fitness += targetReachedReward;
            }
        }
    }
    
    private void OnCollisionEnter(Collision collision)
    {
        if (target != null && collision.gameObject.transform == target) return;
        
        AddReward(crashPenalty * 0.5f); // Use half the crash penalty for non-ground collisions
        EndEpisode();
    }
    
    // --- Heuristic and Helper Methods ---
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetKey(KeyCode.S) ? 1f : (Input.GetKey(KeyCode.W) ? -1f : 0f);
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        continuousActionsOut[2] = Input.GetKey(KeyCode.D) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        continuousActionsOut[3] = Input.GetKey(KeyCode.LeftShift) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }
    
    private void MoveTargetToRandomPosition()
    {
        if (target == null) return;
        Vector3 randomPosition = Random.insideUnitSphere * targetSpawnRadius;
        randomPosition += initialPosition;
        if (randomPosition.y < minSpawnHeight)
        {
            randomPosition.y = minSpawnHeight;
        }
        target.position = randomPosition;
    }

    // --- Genetic Algorithm Methods ---
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
        explorationRate = Mathf.Max(0.1f, explorationRate * explorationDecay);
    }

    private void UpdateGeneticPopulation()
    {
        currentGenes.fitness = totalReward;
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
