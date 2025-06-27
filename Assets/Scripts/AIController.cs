using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIController : MonoBehaviour {
    [SerializeField]
    new Camera camera;
    [SerializeField]
    Plane plane;
    [SerializeField]
    PlaneHUD planeHUD;

    AIInput aiInput;
    PlaneCamera planeCamera;

    void Awake() {
        planeCamera = GetComponent<PlaneCamera>();
        aiInput = GetComponent<AIInput>();
        
        // Add AIInput component if it doesn't exist
        if (aiInput == null) {
            aiInput = gameObject.AddComponent<AIInput>();
        }
    }

    void Start() {
        if (plane != null) {
            SetPlane(plane);
        }
    }

    public void SetPlane(Plane plane) {
        this.plane = plane;

        if (plane != null && planeHUD != null) {
            planeHUD.SetPlane(plane);
            planeHUD.SetCamera(camera);
        }

        planeCamera.SetPlane(plane);
        aiInput.SetPlane(plane);
    }

    void Update() {
        if (plane == null) return;

        // Get inputs from AI system instead of player input
        plane.SetThrottleInput(aiInput.GetThrottleInput());
        plane.SetControlInput(aiInput.GetControlInput());
        planeCamera.SetInput(aiInput.GetCameraInput());
    }
}