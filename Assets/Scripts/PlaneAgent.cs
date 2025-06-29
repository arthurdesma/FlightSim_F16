// PlaneAgent.cs

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

/// <summary>
/// An advanced ML-Agent that learns to fly an aircraft towards a series of dynamically spawning random targets.
/// It must pass through each gate in a stable, forward orientation to get a reward and spawn the next one.
/// It uses a "sonar" system for general terrain avoidance and is heavily rewarded for survival and safe flight.
/// </summary>
public class PlaneAgent : Agent
{
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private Rigidbody planeRigidbody;

    [Header("Dynamic Target Spawning")]
    [Tooltip("The Target Gate prefab to be spawned randomly.")]
    [SerializeField] private GameObject targetGatePrefab;
    [Tooltip("The minimum distance in front of the plane to spawn a new target.")]
    [SerializeField] private float spawnDistanceMin = 400f;
    [Tooltip("The maximum distance in front of the plane to spawn a new target.")]
    [SerializeField] private float spawnDistanceMax = 700f;
    [Tooltip("The radius around the forward spawn point for randomization.")]
    [SerializeField] private float spawnRadius = 300f;
    [Tooltip("The minimum altitude for a new target spawn.")]
    [SerializeField] private float minSpawnHeight = 150f;
    [Tooltip("The maximum altitude for a new target spawn.")]
    [SerializeField] private float maxSpawnHeight = 500f;

    private GameObject currentTargetInstance; // Holds the currently active, spawned gate
    private int score; // Simple counter for how many gates were passed in an episode

    [Header("Episode Settings")]
    [SerializeField] private float maxEpisodeSeconds = 300f;

    [Header("Primary Rewards & Penalties")]
    [SerializeField] private float targetReachedReward = 20.0f;
    [SerializeField] private float crashPenalty = -30.0f;
    [SerializeField] private float timeStepPenalty = -0.001f;
    [SerializeField] private float timeOutPenalty = -10f;
    [SerializeField] private float progressRewardScale = 1.0f;

    [Header("Survival Rewards")]
    [Tooltip("How much to reward the agent for maintaining a stable, level flight.")]
    [SerializeField] private float stabilityScale = 0.2f;
    [Tooltip("How much to reward the agent for staying within a safe altitude range.")]
    [SerializeField] private float safeAltitudeReward = 0.01f;
    [Tooltip("The minimum altitude to receive the safe altitude reward.")]
    [SerializeField] private float safeAltitudeMin = 100f;
    [Tooltip("The altitude below which the agent starts receiving strong penalties.")]
    [SerializeField] private float penaltyAltitude = 75f;
    [Tooltip("The strength of the penalty for flying too low. This should be a large negative number.")]
    [SerializeField] private float groundProximityPenaltyScale = -25f;

    // State tracking variables
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool isReady = false;
    private float lastDistanceToTarget;
    private float episodeTimer;
    private bool isEpisodeActive;

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

        if (targetGatePrefab == null)
        {
            Debug.LogError("Target Gate Prefab has not been assigned in the Inspector!", this.gameObject);
        }
    }

    public override void OnEpisodeBegin()
    {
        isEpisodeActive = true;
        episodeTimer = 0f;
        score = 0;

        if (!isReady) InitializeAgent();

        planeRigidbody.velocity = Vector3.zero;
        planeRigidbody.angularVelocity = Vector3.zero;
        transform.SetPositionAndRotation(initialPosition, initialRotation);

        if (plane != null)
        {
            plane.ResetPlane();
            planeRigidbody.velocity = transform.forward * 40f;
        }

        SpawnNewTargetGate();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!isReady || !isEpisodeActive || plane == null || planeRigidbody == null || currentTargetInstance == null)
        {
            for (int i = 0; i < 22; i++) { sensor.AddObservation(0f); }
            return;
        }

        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.velocity));
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.angularVelocity));

        Vector3 dirToTarget = (currentTargetInstance.transform.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(dirToTarget));
        sensor.AddObservation(Vector3.Distance(transform.position, currentTargetInstance.transform.position) / 1000f);

        sensor.AddObservation(transform.InverseTransformDirection(currentTargetInstance.transform.forward));
        sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up));

        CollectProximitySensorObservations(sensor);
    }
    
    private void CalculateRewards()
    {
        if (!isEpisodeActive || currentTargetInstance == null || plane == null || plane.Dead) return;

        // 1. Survival Rewards (Teach agent to fly safely)
        float stability = Vector3.Dot(transform.up, Vector3.up);
        AddReward(stability * stabilityScale);

        if (transform.position.y > safeAltitudeMin)
        {
            AddReward(safeAltitudeReward);
        }

        // 2. Survival Penalties (Teach agent to fear danger)
        if (transform.position.y < penaltyAltitude)
        {
            float proximityRatio = 1f - (transform.position.y / penaltyAltitude);
            float penalty = proximityRatio * proximityRatio * groundProximityPenaltyScale;
            AddReward(penalty);
        }

        // 3. Objective Rewards (Teach agent to complete its mission)
        float currentDistance = Vector3.Distance(transform.position, currentTargetInstance.transform.position);
        float distanceDelta = lastDistanceToTarget - currentDistance;
        AddReward(distanceDelta * progressRewardScale);
        lastDistanceToTarget = currentDistance;

        AddReward(timeStepPenalty);

        // 4. Terminal State Check (End the episode on a crash)
        if (plane.Dead || transform.position.y < 5f)
        {
            AddReward(crashPenalty);
            TerminateEpisode();
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (plane == null || !isEpisodeActive) return;
        plane.SetControlInput(new Vector3(actions.ContinuousActions[0], actions.ContinuousActions[1], -actions.ContinuousActions[2]));
        plane.SetThrottleInput(actions.ContinuousActions[3]);
        CalculateRewards();
    }

    private void SpawnNewTargetGate()
    {
        if (currentTargetInstance != null) Destroy(currentTargetInstance);
        
        float forwardDistance = Random.Range(spawnDistanceMin, spawnDistanceMax);
        Vector3 randomOffset = Random.insideUnitSphere * spawnRadius;
        Vector3 spawnPosition = transform.position + (transform.forward * forwardDistance) + randomOffset;
        
        spawnPosition.y = Mathf.Clamp(spawnPosition.y, minSpawnHeight, maxSpawnHeight);
        
        Quaternion spawnRotation = Quaternion.LookRotation(transform.position - spawnPosition);
        currentTargetInstance = Instantiate(targetGatePrefab, spawnPosition, spawnRotation);
        
        TargetGate gateComponent = currentTargetInstance.GetComponent<TargetGate>();
        if (gateComponent != null) gateComponent.agent = this;
        
        lastDistanceToTarget = Vector3.Distance(transform.position, currentTargetInstance.transform.position);
    }
    
    public void OnTargetReached()
    {
        if (!isEpisodeActive) return;
        score++;
        AddReward(targetReachedReward);
        SpawnNewTargetGate();
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

    private void OnCollisionEnter(Collision collision)
    {
        // Use Layer check for best performance
        if (collision.gameObject.layer == LayerMask.NameToLayer("Terrain"))
        {
            AddReward(crashPenalty);
            TerminateEpisode();
        }
    }

    private void TerminateEpisode()
    {
        if (isEpisodeActive)
        {
            isEpisodeActive = false;
            if (currentTargetInstance != null)
            {
                Destroy(currentTargetInstance);
            }
            EndEpisode();
        }
    }

    private void CollectProximitySensorObservations(VectorSensor sensor)
    {
        Vector3[] rayDirections = {
            transform.forward, -transform.forward, transform.up, -transform.up,
            transform.right, -transform.right, (transform.forward - transform.up).normalized, (transform.forward + transform.up).normalized
        };
        float sensorMaxDistance = 1000f;
        float sensorSphereRadius = 5f;
        foreach (var dir in rayDirections)
        {
            bool didHit = Physics.SphereCast(transform.position, sensorSphereRadius, dir, out RaycastHit hit, sensorMaxDistance, LayerMask.GetMask("Terrain"));
            if (didHit)
            {
                sensor.AddObservation(hit.distance / sensorMaxDistance);
                Debug.DrawRay(transform.position, dir * hit.distance, Color.yellow);
            }
            else
            {
                sensor.AddObservation(1.0f);
                Debug.DrawRay(transform.position, dir * sensorMaxDistance, Color.cyan);
            }
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
}


// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --force

// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --resume