import json
import paho.mqtt.client as mqtt
import time

MQTT_BROKER = '218.244.158.61'
MQTT_PORT = 1883
MQTT_USERNAME = "tzq"
MQTT_PASSWORD = "123456"
MQTT_PUB_TOPIC = "esp8266_tzq_pub"
MQTT_SUB_TOPIC = "esp8266_tzq_sub"
client_id = "xyz"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully.")
        client.subscribe(MQTT_PUB_TOPIC)
        client.subscribe(MQTT_SUB_TOPIC)

    else:
        print(f"Failed to connect, return code {rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Disconnected from MQTT broker")


def on_subscribe(client, userdata, mid, granted_qos):
    print("On Subscribed: qos = %d" % granted_qos)


def create_mqtt():

    client = mqtt.Client(client_id)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe

    try:

        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        exit(1)

    return client


def send():
    client = create_mqtt()
    client.loop_start()
    time.sleep(3)
    return client


if __name__ == "__main__":
    client = send()

    while True:
        data = {
            "fish_count": 4,       #修改鱼的数目
            "fish_deaths": 0       #死鱼数目
        }
        client.publish(MQTT_PUB_TOPIC, "9" + json.dumps(data))
        time.sleep(2)
