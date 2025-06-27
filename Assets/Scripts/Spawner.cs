using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spawner : MonoBehaviour {
    [SerializeField]
    GameObject planePrefab;
    [SerializeField]
    PlayerController playerController;
    [SerializeField]
    AIController AIController;
    [SerializeField]
    Transform spawnGround;
    [SerializeField]
    Transform spawnAir;

    [SerializeField]
    bool spawnInAir;
    [SerializeField]
    float spawnInAirSpeed;
    
    [SerializeField]
    bool useAI;

    GameObject activePlane;

    public bool SpawnInAir {
        get {
            return spawnInAir;
        }
        set {
            spawnInAir = value;
        }
    }

    void Start() {
        Spawn();
    }

    public void Spawn() {
        Transform spawnPoint = spawnGround;

        if (spawnInAir) {
            spawnPoint = spawnAir;
        }

        Spawn(spawnPoint);
    }

    void Spawn(Transform spawnPoint) {
        // Clean up previous plane
        if (activePlane != null) {
            if (playerController != null) playerController.SetPlane(null);
            if (AIController != null) AIController.SetPlane(null);
            Destroy(activePlane);
        }

        // Spawn new plane
        activePlane = Instantiate(planePrefab);
        Transform t = activePlane.GetComponent<Transform>();
        t.SetPositionAndRotation(spawnPoint.position, spawnPoint.rotation);

        Plane plane = activePlane.GetComponent<Plane>();
        
        // Set the plane to the appropriate controller with null checks
        if (useAI) {
            if (AIController != null) {
                AIController.SetPlane(plane);
            } else {
                Debug.LogError("AI Controller is not assigned in the Spawner!");
            }
        } else {
            if (playerController != null) {
                playerController.SetPlane(plane);
            } else {
                Debug.LogError("Player Controller is not assigned in the Spawner!");
            }
        }
    }
}
