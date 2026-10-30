using UnityEngine;

public class PropellerSpin : MonoBehaviour
{
    public float currentSpeed = 0f;

    void Update()
    {
        transform.Rotate(
            0,
            0,
            currentSpeed * Time.deltaTime,
            Space.Self
        );
    }
}