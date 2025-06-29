// PlaneAgent.cs (Dynamic Spawning Version)

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic; // Still useful for LayerMasks, etc.

/// <summary>
/// An ML-Agent that learns to fly towards a series of dynamically spawning random targets.
/// It must pass through each gate in a stable, forward orientation to get a reward and spawn the next one.
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
    private int score; // Simple counter for how many gates were passed

    [Header("Episode Settings")]
    [SerializeField] private float maxEpisodeSeconds = 300f;

    [Header("Rewards & Penalties")]
    [SerializeField] private float targetReachedReward = 20.0f; // Increased reward for the harder task
    [SerializeField] private float crashPenalty = -30.0f;
    [SerializeField] private float timeStepPenalty = -0.001f;
    [SerializeField] private float timeOutPenalty = -10f;
    [SerializeField] private float progressRewardScale = 1.0f;
    [SerializeField] private float stabilityScale = 0.2f;

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

        // Safety check to ensure the prefab is assigned
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

        // Reset plane physics and position
        planeRigidbody.velocity = Vector3.zero;
        planeRigidbody.angularVelocity = Vector3.zero;
        transform.SetPositionAndRotation(initialPosition, initialRotation);

        if (plane != null)
        {
            plane.ResetPlane();
            planeRigidbody.velocity = transform.forward * 40f;
        }

        // Spawn the very first target for this episode
        SpawnNewTargetGate();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Don't collect observations if the target doesn't exist yet
        if (!isReady || !isEpisodeActive || plane == null || planeRigidbody == null || currentTargetInstance == null) return;

        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.velocity));
        sensor.AddObservation(transform.InverseTransformDirection(planeRigidbody.angularVelocity));

        // Use the spawned gate's transform for observations
        Vector3 dirToTarget = (currentTargetInstance.transform.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(dirToTarget));
        sensor.AddObservation(Vector3.Distance(transform.position, currentTargetInstance.transform.position) / 1000f);

        // Observe the orientation of the gate
        sensor.AddObservation(transform.InverseTransformDirection(currentTargetInstance.transform.forward));
        sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up));

        CollectProximitySensorObservations(sensor);
    }
    
    /// <summary>
    /// The core of the new system. Destroys the old gate and spawns a new one
    /// at a random, reachable position in front of the agent.
    /// </summary>
    private void SpawnNewTargetGate()
    {
        // Clean up the old gate if it exists
        if (currentTargetInstance != null)
        {
            Destroy(currentTargetInstance);
        }

        // Calculate a spawn position
        float forwardDistance = Random.Range(spawnDistanceMin, spawnDistanceMax);
        Vector3 randomOffset = Random.insideUnitSphere * spawnRadius;
        Vector3 spawnPosition = transform.position + (transform.forward * forwardDistance) + randomOffset;
        
        // Clamp the altitude to ensure it's not underground or too high
        spawnPosition.y = Mathf.Clamp(spawnPosition.y, minSpawnHeight, maxSpawnHeight);

        // Set the gate's rotation to face the plane, making it easier to pass through
        Quaternion spawnRotation = Quaternion.LookRotation(transform.position - spawnPosition);

        // Instantiate the new gate
        currentTargetInstance = Instantiate(targetGatePrefab, spawnPosition, spawnRotation);

        // Get the gate's script and tell it about this agent
        TargetGate gateComponent = currentTargetInstance.GetComponent<TargetGate>();
        if (gateComponent != null)
        {
            gateComponent.agent = this;
        }

        // Update the distance tracker
        lastDistanceToTarget = Vector3.Distance(transform.position, currentTargetInstance.transform.position);
    }
    
    /// <summary>
    /// Public method called by the TargetGate script upon a successful pass-through.
    /// </summary>
    public void OnTargetReached()
    {
        if (!isEpisodeActive) return;

        score++;
        AddReward(targetReachedReward);

        // Instead of picking the next gate from a list, we just spawn a new random one.
        SpawnNewTargetGate();
    }

    // --- All other methods (OnActionReceived, FixedUpdate, CalculateRewards, etc.) remain largely the same, ---
    // --- but must refer to `currentTargetInstance.transform` instead of `currentTarget.transform`.         ---

    private void CalculateRewards()
    {
        if (!isEpisodeActive || currentTargetInstance == null || plane == null || plane.Dead) return;

        float stability = Vector3.Dot(transform.up, Vector3.up);
        AddReward(stability * stabilityScale);

        float currentDistance = Vector3.Distance(transform.position, currentTargetInstance.transform.position);
        float distanceDelta = lastDistanceToTarget - currentDistance;
        AddReward(distanceDelta * progressRewardScale);
        lastDistanceToTarget = currentDistance;

        AddReward(timeStepPenalty);

        if (plane.Dead || transform.position.y < 5f) {
            AddReward(crashPenalty);
            TerminateEpisode();
        }
    }

    #region --- Unchanged Methods ---

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (plane == null || !isEpisodeActive) return;
        plane.SetControlInput(new Vector3(actions.ContinuousActions[0], actions.ContinuousActions[1], -actions.ContinuousActions[2]));
        plane.SetThrottleInput(actions.ContinuousActions[3]);
        CalculateRewards();
    }

    private void FixedUpdate()
    {
        if (!isEpisodeActive) return;
        episodeTimer += Time.fixedDeltaTime;
        if (episodeTimer >= maxEpisodeSeconds) {
            AddReward(timeOutPenalty);
            TerminateEpisode();
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Terrain"))
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
            // Clean up the final gate when the episode ends
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
        float sensorMaxDistance = 500f;
        float sensorSphereRadius = 5f;
        foreach (var dir in rayDirections) {
            bool didHit = Physics.SphereCast(transform.position, sensorSphereRadius, dir, out RaycastHit hit, sensorMaxDistance, LayerMask.GetMask("Terrain"));
            if (didHit) {
                sensor.AddObservation(hit.distance / sensorMaxDistance);
                Debug.DrawRay(transform.position, dir * hit.distance, Color.yellow);
            } else {
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

    #endregion
}



// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --force

// mlagents-learn config/flyer_config.yaml --run-id=FirstConnectionTest --resume