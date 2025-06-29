// TargetGate.cs (Slightly Simplified)

using UnityEngine;

public class TargetGate : MonoBehaviour
{
    public PlaneAgent agent;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            Vector3 planeVelocity = other.attachedRigidbody.velocity;
            // Ensure the agent is flying through in the correct forward direction
            if (Vector3.Dot(planeVelocity.normalized, transform.forward) > 0.5f)
            {
                if (agent != null)
                {
                    agent.OnTargetReached();
                }
                // No need to deactivate it, as the agent will destroy it
            }
        }
    }
}
