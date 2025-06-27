using UnityEngine;
using UnityEngine.InputSystem;

// Make sure the filename is "PlayerController.cs"
public class PlayerController : MonoBehaviour {
    [Header("Object References")]
    [SerializeField] private Plane plane;
    [SerializeField] private PlaneHUD planeHUD;
    [SerializeField] private new Camera camera;

    private Vector3 controlInput;
    private PlaneCamera planeCamera;

    void Awake() {
        planeCamera = GetComponent<PlaneCamera>();
    }

    void Start() {
        if (plane == null) {
            Debug.LogError("Plane is not assigned in the PlayerController Inspector!", this);
            return;
        }

        if (planeHUD != null) {
            planeHUD.SetPlane(plane);
            planeHUD.SetCamera(camera);
        }

        if (planeCamera != null) {
            planeCamera.SetPlane(plane);
        } else {
             Debug.LogError("PlaneCamera component could not be found on the same GameObject!", this);
        }
    }
    
    // ---- INPUT SYSTEM METHODS ----
    // ---- ADD 'public' TO ALL OF THESE ----

    public void SetThrottleInput(InputAction.CallbackContext context) {
        if (plane == null) return;
        plane.SetThrottleInput(context.ReadValue<float>());
    }

    public void OnRollPitchInput(InputAction.CallbackContext context) {
        if (plane == null) return;
        var input = context.ReadValue<Vector2>();
        controlInput = new Vector3(input.y, controlInput.y, -input.x);
    }

    public void OnYawInput(InputAction.CallbackContext context) {
        if (plane == null) return;
        var input = context.ReadValue<float>();
        controlInput = new Vector3(controlInput.x, input, controlInput.z);
    }

    public void OnCameraInput(InputAction.CallbackContext context) {
        if (plane == null || planeCamera == null) return;
        var input = context.ReadValue<Vector2>();
        planeCamera.SetInput(input);
    }

    public void OnToggleHelp(InputAction.CallbackContext context) {
        if (plane == null || planeHUD == null) return;
        if (context.phase == InputActionPhase.Performed) {
            planeHUD.ToggleHelpDialogs();
        }
    }

    // ---- UPDATE METHOD ----
    void Update() {
        if (plane == null) return;
        plane.SetControlInput(controlInput);
    }
}
