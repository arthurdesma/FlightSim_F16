using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// An ML-Agent that learns to fly a plane towards targets while avoiding crashes.
/// Simplified and optimized for better crash avoidance learning.
/// </summary>
public class PlaneAgent : Agent
{
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private Rigidbody planeRigidbody;
    [SerializeField] private Transform target;

    [Header("Episode Settings")]
    [SerializeField] private float targetSpawnRadius = 300f;
    [SerializeField] private float minSpawnHeight = 100f; // Increased for safety
    [SerializeField] private float targetReachedRadius = 40f;
    [SerializeField] private float maxEpisodeSeconds = 120f; // Shorter episodes

    [Header("Primary Rewards & Penalties")]
    [SerializeField] private float targetReachedReward = 15.0f;
    [SerializeField] private float crashPenalty = -30.0f; // Increased penalty
    [SerializeField] private float timeStepPenalty = -0.001f; // Reduced
    [SerializeField] private float timeOutPenalty = -10f;

    [Header("Safety Rewards")]
    [SerializeField] private float progressRewardScale = 1.0f;
    [SerializeField] private float stabilityScale = 0.2f;
    [SerializeField] private float altitudeRewardScale = 0.5f;
    [SerializeField] private float groundProximityPenaltyScale = -15.0f; // Increased
    [SerializeField] private float terrainProximityPenaltyScale = -20f; // Increased

    [Header("Terrain Avoidance")]
    [SerializeField] private float terrainCheckDistance = 100f;
    [SerializeField] private float terrainCheckSphereRadius = 10f;

    [Header("Flight Parameters")]
    [SerializeField] private float optimalSpeed = 60.0f; // Reduced for stability
    [SerializeField] private float stallAngleThreshold = 15.0f;
    [SerializeField] private float maxAllowedAngularVelocity = 1.5f;

    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool isReady = false;
    
    // State tracking variables
    private float lastDistanceToTarget;
    private float episodeTimer;
    private bool isEpisodeActive;
    private float totalReward = 0f;

    // --- INITIALIZATION & EPISODE MANAGEMENT ---

    public override void Initialize()
    {
        InitializeAgent();
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
        isEpisodeActive = true;
        totalReward = 0f;
        episodeTimer = 0f;
        
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
            // Gentler initial velocity for learning
            planeRigidbody.velocity = transform.forward * 30f;
        }

        MoveTargetToRandomPosition();
        
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

        // Velocity and angular velocity in local space
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.velocity)); // 3
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.angularVelocity)); // 3

        // Direction and distance to target
        Vector3 dirToTarget = (target.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(dirToTarget)); // 3
        float currentDistance = Vector3.Distance(transform.position, target.position);
        sensor.AddObservation(currentDistance / 1000f); // Normalized distance // 1

        // Plane state
        sensor.AddObservation(plane.Throttle); // 1
        sensor.AddObservation(plane.AngleOfAttack); // 1
        
        // Stability (how upright the plane is)
        sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up)); // 1

        // Ground proximity (critical for crash avoidance)
        float groundDistance = transform.position.y;
        sensor.AddObservation(groundDistance / 200f); // Normalized ground distance // 1

        // Remaining time
        float remainingTimeNormalized = Mathf.Max(0, 1f - (episodeTimer / maxEpisodeSeconds));
        sensor.AddObservation(remainingTimeNormalized); // 1

        // Total: 14 observations
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (plane == null || !isEpisodeActive) return;

        float pitch = actions.ContinuousActions[0];
        float yaw = actions.ContinuousActions[1];
        float roll = actions.ContinuousActions[2];
        float throttle = actions.ContinuousActions[3];

        plane.SetControlInput(new Vector3(pitch, yaw, -roll));
        plane.SetThrottleInput(throttle);

        CalculateRewards();
    }
    
    private void FixedUpdate()
    {
        if (!isEpisodeActive) return;
        
        episodeTimer += Time.fixedDeltaTime;
        if (episodeTimer >= maxEpisodeSeconds)
        {
            AddReward(timeOutPenalty);
            TerminateEpisode();
        }
    }

    // --- SIMPLIFIED REWARD CALCULATION FOCUSED ON CRASH AVOIDANCE ---

    private void CalculateRewards()
    {
        if (!isEpisodeActive || target == null || plane == null || plane.Dead) return;

        // === CRITICAL: CRASH AVOIDANCE REWARDS ===
        
        // 1. Ground proximity penalty (early warning system)
        float groundHeight = transform.position.y;
        if (groundHeight < 50f)
        {
            float proximityRatio = (50f - groundHeight) / 50f;
            float penalty = -proximityRatio * proximityRatio * groundProximityPenaltyScale;
            AddReward(penalty);
        }

        // 2. Terrain avoidance
        CalculateTerrainAvoidancePenalty();

        // 3. Stability reward (prevent spinning out of control)
        float stability = Vector3.Dot(transform.up, Vector3.up);
        AddReward(stability * stabilityScale);

        // 4. Altitude maintenance reward
        if (groundHeight > 75f && groundHeight < 300f) // Sweet spot altitude
        {
            AddReward(altitudeRewardScale);
        }

        // === PROGRESS TOWARDS GOAL ===
        
        // 5. Progress reward
        float currentDistance = Vector3.Distance(transform.position, target.position);
        float distanceDelta = lastDistanceToTarget - currentDistance;
        if (distanceDelta > 0)
        {
            AddReward(distanceDelta * progressRewardScale);
        }
        lastDistanceToTarget = currentDistance;

        // 6. Small time penalty to encourage efficiency
        AddReward(timeStepPenalty);

        // === TARGET REACHED ===
        if (currentDistance < targetReachedRadius)
        {
            AddReward(targetReachedReward);
            MoveTargetToRandomPosition();
            lastDistanceToTarget = Vector3.Distance(transform.position, target.position);
        }
        
        totalReward = GetCumulativeReward();

        // === EPISODE END CONDITIONS ===
        if (plane.Dead || transform.position.y < 5f) // Crash conditions
        {
            AddReward(crashPenalty); // FIXED: Using AddReward instead of SetReward
            TerminateEpisode();
        }
    }
    
    private void CalculateTerrainAvoidancePenalty()
    {
        Vector3 checkDirection = planeRigidbody.velocity.magnitude > 5f ? 
            planeRigidbody.velocity.normalized : transform.forward;
            
        if (Physics.SphereCast(transform.position, terrainCheckSphereRadius, 
            checkDirection, out RaycastHit hit, terrainCheckDistance))
        {
            if (hit.collider.CompareTag("Terrain"))
            {
                float proximityRatio = 1f - (hit.distance / terrainCheckDistance);
                float penalty = proximityRatio * proximityRatio * terrainProximityPenaltyScale;
                AddReward(penalty);
            }
        }
    }
    
    private void OnCollisionEnter(Collision collision)
    {
        // Ignore collisions with the target
        if (target != null && collision.gameObject.transform == target) return;
        
        // Major crash penalty
        AddReward(crashPenalty); // FIXED: Using AddReward
        TerminateEpisode();
    }
    
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
        
        // Pitch (W/S)
        continuousActionsOut[0] = Input.GetKey(KeyCode.S) ? 1f : (Input.GetKey(KeyCode.W) ? -1f : 0f);
        // Yaw (Q/E)
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        // Roll (A/D)
        continuousActionsOut[2] = Input.GetKey(KeyCode.D) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        // Throttle (Shift/Ctrl)
        continuousActionsOut[3] = Input.GetKey(KeyCode.LeftShift) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }

    private void MoveTargetToRandomPosition()
    {
        if (target == null) return;

        // Safety check to prevent infinite loop
        if (targetSpawnRadius <= targetReachedRadius)
        {
            Debug.LogError("targetSpawnRadius must be greater than targetReachedRadius");
            target.position = transform.position + (Vector3.up * 100f) + (transform.forward * 100f);
            return;
        }

        Vector3 randomPosition;
        int maxAttempts = 50;

        for (int i = 0; i < maxAttempts; i++)
        {
            // Get a random point within a sphere around the agent's current position
            randomPosition = transform.position + (Random.insideUnitSphere * targetSpawnRadius);
            
            // Ensure the target is at a safe height
            randomPosition.y = Mathf.Max(randomPosition.y, minSpawnHeight);
            
            // Make sure it's not too close to terrain
            randomPosition.y = Mathf.Min(randomPosition.y, 400f);

            // Check if the position is valid (far enough from current position)
            if (Vector3.Distance(transform.position, randomPosition) > targetReachedRadius)
            {
                target.position = randomPosition;
                return;
            }
        }
        
        // Fallback position
        Debug.LogWarning("Could not find valid target position, using fallback");
        target.position = transform.position + (transform.forward * (targetSpawnRadius * 0.7f)) + (Vector3.up * minSpawnHeight);
    }
}

// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --force

// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --resume