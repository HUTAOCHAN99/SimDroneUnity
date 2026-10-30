using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -5);
    public float smoothSpeed = 5f;

    void LateUpdate()
    {
        if (target == null) return;

        Quaternion yawOnly =
            Quaternion.Euler(
                0,
                target.eulerAngles.y,
                0
            );

        Vector3 desiredPosition =
            target.position +
            yawOnly * offset;

        transform.position = Vector3.Lerp(
            transform.position,
            desiredPosition,
            smoothSpeed * Time.deltaTime
        );

        transform.LookAt(target.position + Vector3.up);
    }
}