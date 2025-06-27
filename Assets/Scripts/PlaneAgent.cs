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

    [Header("Genetic Algorithm Enhancement")]
    [SerializeField] private bool useGeneticBootstrap = true;
    [SerializeField] private float mutationRate = 0.1f;
    [SerializeField] private float explorationDecay = 0.995f;
    
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool isReady = false;
    
    // Performance tracking
    private float episodeStartTime;
    private float bestDistance = float.MaxValue;
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
        public float[] genes; // Store successful action patterns
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

    private void Awake()
    {
        Debug.Log($"[PlaneAgent] Awake - GameObject: {gameObject.name}");
    }

    private void Start()
    {
        Debug.Log($"[PlaneAgent] Start - Agent enabled: {enabled}");
        
        if (!isReady)
        {
            InitializeAgent();
        }
    }

    public override void Initialize()
    {
        Debug.Log($"[PlaneAgent] Initialize called by ML-Agents");
        InitializeAgent();
        
        // Initialize genetic memory if needed
        if (useGeneticBootstrap && populationMemory.Count == 0)
        {
            for (int i = 0; i < 10; i++)
            {
                populationMemory.Add(new GeneticMemory(8)); // 4 actions * 2 (base + modifier)
            }
        }
        
        currentGenes = new GeneticMemory(8);
    }

    private void InitializeAgent()
    {
        if (isReady) return;
        
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        
        if (plane == null) plane = GetComponent<Plane>();
        if (planeRigidbody == null) planeRigidbody = GetComponent<Rigidbody>();
        
        isReady = true;
        
        Debug.Log($"[PlaneAgent] InitializeAgent complete - Plane: {plane != null}, Rigidbody: {planeRigidbody != null}, Target: {target != null}");
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log($"[PlaneAgent] OnEpisodeBegin - Previous episode reward: {totalReward}");
        
        // Update genetic algorithm
        if (useGeneticBootstrap && totalReward != 0)
        {
            UpdateGeneticPopulation();
        }
        
        // Reset episode variables
        episodeStartTime = Time.time;
        bestDistance = float.MaxValue;
        totalReward = 0f;
        successfulTargets = 0;
        
        if (!isReady) InitializeAgent();
        
        // Reset physics
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
            // Give initial velocity for easier start
            planeRigidbody.velocity = transform.forward * 50f;
        }
        
        MoveTargetToRandomPosition();
        
        // Select genes for this episode
        if (useGeneticBootstrap)
        {
            SelectGenesForEpisode();
        }
    }

    private void SelectGenesForEpisode()
    {
        if (Random.value < explorationRate)
        {
            // Exploration: use random genes
            currentGenes = new GeneticMemory(8);
        }
        else
        {
            // Exploitation: use best genes with mutation
            var bestGenes = populationMemory.OrderByDescending(g => g.fitness).First();
            currentGenes = bestGenes.Clone();
            
            // Apply mutation
            for (int i = 0; i < currentGenes.genes.Length; i++)
            {
                if (Random.value < mutationRate)
                {
                    currentGenes.genes[i] += Random.Range(-0.3f, 0.3f);
                    currentGenes.genes[i] = Mathf.Clamp(currentGenes.genes[i], -1f, 1f);
                }
            }
        }
        
        // Decay exploration rate
        explorationRate *= explorationDecay;
        explorationRate = Mathf.Max(explorationRate, 0.1f); // Keep minimum exploration
    }

    private void UpdateGeneticPopulation()
    {
        currentGenes.fitness = totalReward + (successfulTargets * 10f);
        currentGenes.avgSpeed = plane.LocalVelocity.magnitude;
        currentGenes.avgAltitude = plane.AltitudeFeet;
        
        // Replace worst performer
        var worstIndex = 0;
        var worstFitness = float.MaxValue;
        
        for (int i = 0; i < populationMemory.Count; i++)
        {
            if (populationMemory[i].fitness < worstFitness)
            {
                worstFitness = populationMemory[i].fitness;
                worstIndex = i;
            }
        }
        
        if (currentGenes.fitness > worstFitness)
        {
            populationMemory[worstIndex] = currentGenes.Clone();
            Debug.Log($"[PlaneAgent] Updated genetic population. New fitness: {currentGenes.fitness}");
        }
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

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!isReady) InitializeAgent();
        
        // Enhanced observations for better learning
        
        // 1. Plane orientation (3 values) - using euler angles normalized
        Vector3 euler = transform.rotation.eulerAngles;
        sensor.AddObservation(Mathf.Sin(euler.x * Mathf.Deg2Rad));
        sensor.AddObservation(Mathf.Sin(euler.y * Mathf.Deg2Rad));
        sensor.AddObservation(Mathf.Sin(euler.z * Mathf.Deg2Rad));
        
        // 2. Velocity information (3 values)
        if (plane != null)
        {
            // Use local velocity for better understanding of movement
            Vector3 localVel = plane.LocalVelocity / 100f; // Normalize to reasonable range
            sensor.AddObservation(localVel.x);
            sensor.AddObservation(localVel.y);
            sensor.AddObservation(localVel.z);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }
        
        // 3. Target information (3 values)
        if (target != null)
        {
            Vector3 dirToTarget = (target.position - transform.position).normalized;
            sensor.AddObservation(dirToTarget);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }
        
        // 4. Flight dynamics (2 values)
        if (plane != null)
        {
            sensor.AddObservation(plane.AngleOfAttack / Mathf.PI); // Normalized AOA
            sensor.AddObservation(plane.Throttle); // Current throttle
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
        
        // Total: 3 + 3 + 3 + 2 = 11 observations
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (plane == null) return;
        
        float pitch = actions.ContinuousActions[0];
        float yaw = actions.ContinuousActions[1];
        float roll = actions.ContinuousActions[2];
        float throttle = actions.ContinuousActions[3];
        
        // Apply genetic influence if enabled
        if (useGeneticBootstrap && currentGenes != null)
        {
            float influence = Mathf.Max(0.2f, explorationRate);
            pitch = Mathf.Lerp(pitch, currentGenes.genes[0], influence);
            yaw = Mathf.Lerp(yaw, currentGenes.genes[1], influence);
            roll = Mathf.Lerp(roll, currentGenes.genes[2], influence);
            throttle = Mathf.Lerp(throttle, currentGenes.genes[3], influence);
        }
        
        // Apply controls
        plane.SetControlInput(new Vector3(pitch, yaw, -roll));
        plane.SetThrottleInput(throttle);
        
        // Calculate rewards
        CalculateRewards();
    }
    
    private void CalculateRewards()
    {
        if (target == null || plane == null) return;
        
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        Vector3 dirToTarget = (target.position - transform.position).normalized;
        
        // 1. Distance reward (encourage getting closer)
        if (distanceToTarget < bestDistance)
        {
            float improvement = bestDistance - distanceToTarget;
            AddReward(0.001f * improvement);
            bestDistance = distanceToTarget;
        }
        
        // 2. Alignment reward
        float alignment = Vector3.Dot(transform.forward, dirToTarget);
        AddReward(0.002f * alignment);
        
        // 3. Stable flight rewards
        float levelFlight = Vector3.Dot(transform.up, Vector3.up);
        AddReward(0.001f * levelFlight);
        
        // 4. Speed maintenance reward
        float speedReward = Mathf.Clamp01(plane.LocalVelocity.z / 100f);
        AddReward(0.0005f * speedReward);
        
        // 5. Altitude penalty (don't fly too low)
        if (transform.position.y < minSpawnHeight * 0.5f)
        {
            AddReward(-0.01f);
        }
        
        // 6. Stall penalty
        if (Mathf.Abs(plane.AngleOfAttack) > 0.3f)
        {
            AddReward(-0.005f);
        }
        
        // Time penalty
        AddReward(-0.0001f);
        
        // Track total reward
        totalReward = GetCumulativeReward();
        
        // End episode if plane crashes
        if (plane.Dead || transform.position.y < 0)
        {
            SetReward(-10f);
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetKey(KeyCode.S) ? 1f : (Input.GetKey(KeyCode.W) ? -1f : 0f);
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        continuousActionsOut[2] = Input.GetKey(KeyCode.D) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        continuousActionsOut[3] = Input.GetKey(KeyCode.LeftShift) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }
    
    private void OnCollisionEnter(Collision collision)
    {
        if (target != null && collision.gameObject.transform == target) return;
        
        SetReward(-5f);
        EndEpisode();
    }
    
    private void OnTriggerEnter(Collider other)
    {
        if (target != null && other.transform == target)
        {
            successfulTargets++;
            AddReward(10f);
            MoveTargetToRandomPosition();
            
            // Store successful pattern
            if (useGeneticBootstrap && currentGenes != null)
            {
                currentGenes.fitness += 5f;
            }
        }
    }
}
