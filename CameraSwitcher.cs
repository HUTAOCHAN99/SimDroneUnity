using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera fpvCamera;
    public Camera thirdPersonCamera;

    void Start()
    {
        if (fpvCamera != null) fpvCamera.enabled = false;
        if (thirdPersonCamera != null) thirdPersonCamera.enabled = true;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            if (fpvCamera != null && thirdPersonCamera != null)
            {
                fpvCamera.enabled = !fpvCamera.enabled;
                thirdPersonCamera.enabled = !thirdPersonCamera.enabled;
            }
        }
    }
}