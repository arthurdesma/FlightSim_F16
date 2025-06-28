// SphericalTagDetector.cs

using UnityEngine;

/// <summary>
/// Casts a sphere of rays from the object to detect a specific tag.
/// The rays are visualized in the Scene view when Gizmos are enabled.
/// </summary>
public class SphericalTagDetector : MonoBehaviour
{
    [Header("Raycast Configuration")]
    [Tooltip("The tag of objects to detect (e.g., 'Terrain', 'Obstacle').")]
    [SerializeField] private string detectableTag = "Terrain";

    [Tooltip("The total number of rays to cast in a sphere.")]
    [SerializeField] private int numberOfRays = 100;

    [Tooltip("How far the rays will detect objects.")]
    [SerializeField] private float rayDistance = 50f;

    [Tooltip("The thickness of the rays. > 0 is better for not missing small objects.")]
    [SerializeField] private float rayRadius = 0.1f;

    [Header("Ray Visualization (In Scene View)")]
    [Tooltip("The color of the ray when it does not hit a detectable object.")]
    [SerializeField] private Color noHitColor = Color.green;

    [Tooltip("The color of the ray when it hits a detectable object.")]
    [SerializeField] private Color hitColor = Color.red;


    /// <summary>
    /// Public property to check if any of the rays have detected an obstacle.
    /// </summary>
    public bool IsObstacleDetected { get; private set; }


    // Using FixedUpdate is better for physics-related code like raycasting.
    void FixedUpdate()
    {
        // Reset the detection flag at the start of each physics step.
        // It will be set to true if any ray finds an obstacle.
        IsObstacleDetected = false; 
        
        CastSphericalRays();
    }

    private void CastSphericalRays()
    {
        Vector3 origin = transform.position;

        // --- Fibonacci Sphere Algorithm ---
        // This creates evenly distributed points on a sphere.
        float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));

        for (int i = 0; i < numberOfRays; i++)
        {
            float y = 1 - (i / (float)(numberOfRays - 1)) * 2; // y goes from 1 to -1
            float radius = Mathf.Sqrt(1 - y * y); // Radius at this y-height

            float theta = goldenAngle * i; // The angle for this point

            float x = Mathf.Cos(theta) * radius;
            float z = Mathf.Sin(theta) * radius;

            Vector3 worldDirection = transform.TransformDirection(new Vector3(x, y, z).normalized);

            RaycastHit hitInfo;

            // Using SphereCast is slightly more expensive but much more reliable than a thin Raycast line.
            bool didHit = Physics.SphereCast(origin, rayRadius, worldDirection, out hitInfo, rayDistance);

            if (didHit && hitInfo.collider.CompareTag(detectableTag))
            {
                // HIT! The ray hit an object AND it has the correct tag.
                IsObstacleDetected = true;

                // To make the ray visible, draw a red line from the origin to the hit point.
                Debug.DrawRay(origin, worldDirection * hitInfo.distance, hitColor);
            }
            else
            {
                // MISS! The ray either hit nothing or hit an object with the wrong tag.
                
                // To make the ray visible, draw a green line for the full length.
                Debug.DrawRay(origin, worldDirection * rayDistance, noHitColor);
            }
        }
    }
}
