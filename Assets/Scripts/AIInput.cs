using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

// The AIInput script now IS the Agent, handling both ML and plane communication.
[RequireComponent(typeof(Plane))]
[RequireComponent(typeof(Rigidbody))]
public class AIInput : Agent
{
    [Header("Training Goal: Survival")]
    [Tooltip("The agent will be penalized and the episode will end if it goes below this altitude.")]
    [SerializeField] private float crashAltitudeThreshold = 2f;
    [Tooltip("The agent will be penalized and the episode will end if it flies above this altitude.")]
    [SerializeField] private float maxAltitude = 400f;
    [Tooltip("How much to reward the agent just for staying alive each step.")]
    [SerializeField] private float rewardForLiving = 0.01f;

    // These fields are now controlled by the AI and read by the Plane.cs script
    private Vector3 controlInput;
    private float throttleInput;
    private Vector2 cameraInput; // Unused by this AI, but kept for compatibility

    private Rigidbody rb;
    private Vector3 startingPosition;
    private Quaternion startingRotation;

    // --- ML-Agents Core Methods ---

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startingPosition = transform.position;
        startingRotation = transform.rotation;
    }

    /// <summary>
    /// Resets the plane at the start of each training episode.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // Reset physics
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Reset position and orientation
        transform.position = startingPosition;
        transform.rotation = startingRotation;

        // Reset controls
        controlInput = Vector3.zero;
        throttleInput = 0f;
    }

    /// <summary>
    /// The AI's "senses". We provide it with data about its state.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // --- 11 Observations ---

        // Orientation: Is the plane upside down? (3 observations)
        sensor.AddObservation(transform.up);

        // Velocity: Which direction and how fast is it moving? (3 observations)
        sensor.AddObservation(rb.velocity.normalized);

        // Altitude: How high is it? (1 observation)
        sensor.AddObservation(transform.position.y);

        // Proximity to ground (Raycast): A direct sense of danger below. (1 observation)
        if (Physics.Raycast(transform.position, Vector3.down, out RaycastHit hit, maxAltitude))
        {
            sensor.AddObservation(hit.distance / maxAltitude); // Normalize distance
        }
        else
        {
            sensor.AddObservation(1.0f); // Max distance
        }
        
        // Previous inputs (3 observations)
        sensor.AddObservation(controlInput);
    }

    /// <summary>
    /// The AI's "actions". It receives instructions from the PPO model.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // --- Receive Actions from the PPO Model ---
        // The model outputs 4 continuous values between -1 and 1.
        float pitch = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float yaw = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float roll = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float throttle = actions.ContinuousActions[3]; // Raw value is [-1, 1]

        // --- Apply Actions ---
        // We update the private variables that the public getters expose to Plane.cs.
        controlInput = new Vector3(pitch, yaw, roll);
        throttleInput = (throttle + 1f) / 2f; // Remap throttle from [-1, 1] to [0, 1]

        // --- Reward Logic ---
        // This is where the AI learns what is "good" and "bad".
        
        // 1. Reward for staying alive and level.
        AddReward(rewardForLiving);
        AddReward(Vector3.Dot(transform.up, Vector3.up) * 0.005f); // Reward for being upright

        // --- End Conditions ---
        // 2. Penalize and end the episode for crashing or flying too high.
        if (transform.position.y < crashAltitudeThreshold || transform.position.y > maxAltitude)
        {
            SetReward(-1.0f); // Negative reward for failure
            EndEpisode();
        }
    }
    
    /// <summary>
    /// Allows you to control the plane with a keyboard for testing.
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions.Clear();

        // Pitch (W/S), Yaw (A/D), Roll (Q/E)
        continuousActions[0] = Input.GetAxis("Vertical");
        continuousActions[1] = Input.GetAxis("Horizontal"); 
        continuousActions[2] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.Q) ? -1f : 0f);
        
        // Remap throttle from [0, 1] to [-1, 1] for the heuristic action
        float currentThrottle = Input.GetKey(KeyCode.Space) ? 1f : 0f;
        continuousActions[3] = (currentThrottle * 2f) - 1f;
    }


    // --- PUBLIC GETTERS ---
    // These methods are unchanged. Your Plane.cs will call these just like before.
    public float GetThrottleInput() => throttleInput;
    public Vector3 GetControlInput() => controlInput;
    public Vector2 GetCameraInput() => cameraInput;
}
