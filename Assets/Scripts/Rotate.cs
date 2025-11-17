using UnityEngine;

public class Rotate : MonoBehaviour
{
    public Vector3 rotationSpeed = new Vector3(0f, 20f, 0f);
    void Start(){}
    void Update()
    {
        transform.Rotate(rotationSpeed * Time.deltaTime);
    }
}
