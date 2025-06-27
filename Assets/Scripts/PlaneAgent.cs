// Filename: PlaneAgent.cs
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class PlaneAgent : Agent
{
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private Rigidbody planeRigidbody;

    [Header("Training")]
    [Tooltip("The target the plane should fly towards.")]
    [SerializeField] private Transform target;
    
    [Tooltip("The radius around the starting point where the target will randomly spawn.")]
    [SerializeField] private float targetSpawnRadius = 500f;

    [Tooltip("The minimum height the target can spawn at.")]
    [SerializeField] private float minSpawnHeight = 50f;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public override void Initialize()
    {
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        
        if (plane == null) plane = GetComponent<Plane>();
        if (planeRigidbody == null) planeRigidbody = GetComponent<Rigidbody>();
    }

    /// <summary>
    /// MODIFICATION: Public method to allow external scripts to assign a new target.
    /// This makes the agent reusable for different tasks or waypoints.
    /// </summary>
    /// <param name="newTarget">The new transform for the agent to target.</param>
    public void SetTarget(Transform newTarget)
    {
        target = newTarget;
    }

    // This is called at the beginning of each training episode.
    public override void OnEpisodeBegin()
    {
        // Reset the plane's physics and position
        planeRigidbody.velocity = Vector3.zero;
        planeRigidbody.angularVelocity = Vector3.zero;
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        
        // Assuming you have a reset method in your Plane class
        if (plane != null) {
            plane.ResetPlane(); 
        }

        // MODIFICATION: Move the target to a new random position for each episode.
        MoveTargetToRandomPosition();
    }

    /// <summary>
    /// MODIFICATION: New method to handle repositioning the target.
    /// This keeps the code clean and reusable.
    /// </summary>
    private void MoveTargetToRandomPosition()
    {
        if (target == null) return;
        
        // Generate a random position within a sphere.
        Vector3 randomPosition = Random.insideUnitSphere * targetSpawnRadius;
        randomPosition += initialPosition; // Center the sphere on the agent's start point.
        
        // Ensure the target doesn't spawn below a certain height.
        if(randomPosition.y < minSpawnHeight)
        {
            randomPosition.y = minSpawnHeight;
        }

        target.position = randomPosition;
    }


    // This is where the Agent "sees" the world.
    public override void CollectObservations(VectorSensor sensor)
    {
        if (target == null)
        {
            // If the target is missing, add zeroed-out observations
            // to prevent errors, but this state is not ideal for training.
            sensor.AddObservation(Quaternion.identity); // 4 zeros
            sensor.AddObservation(Vector3.zero); // 3 zeros
            sensor.AddObservation(Vector3.zero); // 3 zeros
            sensor.AddObservation(0f); // 1 zero
            return;
        }

        // 1. Observe the plane's orientation. (4 observations)
        sensor.AddObservation(transform.rotation);

        // 2. Observe the plane's velocity. (3 observations)
        sensor.AddObservation(planeRigidbody.velocity.normalized);
        
        // 3. Observe the direction to the target. (3 observations)
        Vector3 directionToTarget = (target.position - transform.position).normalized;
        sensor.AddObservation(directionToTarget);

        // 4. Observe the alignment with the target. (1 observation)
        float alignment = Vector3.Dot(transform.forward, directionToTarget);
        sensor.AddObservation(alignment);

        // Total Observations: 11
    }

    // This is called when the Agent receives an action from the PPO model.
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Map continuous actions to plane controls
        var pitch = actions.ContinuousActions[0];
        var yaw = actions.ContinuousActions[1];
        var roll = -actions.ContinuousActions[2];
        var controlInput = new Vector3(pitch, yaw, roll);
        plane.SetControlInput(controlInput);

        var throttleInput = (actions.ContinuousActions[3] + 1f) / 2f;
        plane.SetThrottleInput(throttleInput);

        // --- REWARD FUNCTION ---
        if(target != null)
        {
            // Reward for facing the target
            Vector3 directionToTarget = (target.position - transform.position).normalized;
            float alignment = Vector3.Dot(transform.forward, directionToTarget);
            AddReward(0.001f * alignment);

            // Reward for being right-side up
            float upsideDownPenalty = Mathf.Clamp01(1 - Vector3.Dot(transform.up, Vector3.up));
            AddReward(-0.002f * upsideDownPenalty);
        }

        // Small penalty per step to encourage efficiency.
        AddReward(-0.0001f);
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions.Clear();

        continuousActions[0] = Input.GetKey(KeyCode.S) ? 1f : (Input.GetKey(KeyCode.W) ? -1f : 0f);
        continuousActions[1] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        continuousActions[2] = Input.GetKey(KeyCode.D) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        continuousActions[3] = Input.GetKey(KeyCode.LeftShift) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }
    
    private void OnCollisionEnter(Collision collision)
    {
        // Don't penalize for colliding with the target trigger.
        if (target != null && collision.gameObject.transform == target)
        {
            return;
        }
        
        // Large penalty for crashing into anything else.
        AddReward(-1.0f);
        EndEpisode();
    }
    
    private void OnTriggerEnter(Collider other)
    {
        if (target != null && other.transform == target)
        {
            // MODIFICATION: Instead of ending the episode, we give a large
            // reward and move the target to a new position. This teaches
            // the agent to fly continuously.
            AddReward(1.0f);
            MoveTargetToRandomPosition();
        }
    }
}
