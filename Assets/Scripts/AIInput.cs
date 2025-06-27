using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIInput : MonoBehaviour {
    [SerializeField]
    float throttleInput = 0.5f;
    [SerializeField]
    Vector3 controlInput;
    [SerializeField]
    Vector2 cameraInput;
    
    // AI behavior parameters
    [SerializeField]
    float cruiseAltitude = 100f;
    [SerializeField]
    float cruiseSpeed = 50f;
    [SerializeField]
    float turnRadius = 50f;
    [SerializeField]
    float inputSmoothness = 2f;
    
    private Plane plane;
    private Vector3 targetControlInput;
    private float targetThrottleInput;
    private Vector2 targetCameraInput;
    
    public void SetPlane(Plane plane) {
        this.plane = plane;
    }
    
    void Update() {
        if (plane == null) return;
        
        CalculateAIInputs();
        SmoothInputs();
    }
    
    void CalculateAIInputs() {
        // Simple AI logic - you can expand this
        Transform planeTransform = plane.transform;
        
        // Throttle control based on desired speed
        float currentSpeed = plane.Velocity.magnitude;
        if (currentSpeed < cruiseSpeed) {
            targetThrottleInput = 1f;
        } else {
            targetThrottleInput = 0.5f;
        }
        
        // Altitude control
        float currentAltitude = planeTransform.position.y;
        float altitudeDifference = cruiseAltitude - currentAltitude;
        
        // Pitch control for altitude
        float pitchInput = Mathf.Clamp(altitudeDifference * 0.01f, -1f, 1f);
        
        // Simple circular flight pattern
        float time = Time.time * 0.1f;
        float rollInput = Mathf.Sin(time) * 0.3f;
        float yawInput = Mathf.Cos(time) * 0.2f;
        
        targetControlInput = new Vector3(pitchInput, yawInput, rollInput);
        
        // Camera movement (optional - can be random or follow a pattern)
        targetCameraInput = new Vector2(
            Mathf.Sin(time * 0.5f) * 0.2f,
            Mathf.Cos(time * 0.3f) * 0.1f
        );
    }
    
    void SmoothInputs() {
        // Smooth the inputs for more realistic AI behavior
        controlInput = Vector3.Lerp(controlInput, targetControlInput, Time.deltaTime * inputSmoothness);
        throttleInput = Mathf.Lerp(throttleInput, targetThrottleInput, Time.deltaTime * inputSmoothness);
        cameraInput = Vector2.Lerp(cameraInput, targetCameraInput, Time.deltaTime * inputSmoothness);
    }
    
    // Public getters for the controller to access
    public float GetThrottleInput() => throttleInput;
    public Vector3 GetControlInput() => controlInput;
    public Vector2 GetCameraInput() => cameraInput;
}
