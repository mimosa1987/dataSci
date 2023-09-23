import traceback
from rocketmq.client import Producer, Message
import json


class RocketMQ(object):
    """

    """

    def __init__(self, url, group_id, **kwargs):
        access_key = kwargs.get('access_key')
        access_secret = kwargs.get('access_secret')

        self.producer = Producer(group_id)

        if access_key is not None and access_secret is not None:
            self.producer.set_session_credentials(access_key=access_key,
                                                  access_secret=access_secret,
                                                  channel='ALIYUN')

        self.producer.set_name_server_address(url)

        lazy_load = kwargs.get('lazy_load')
        if not lazy_load:
            self.producer.start()

        topic = kwargs.get('topic')
        if topic:
            self.msg = Message(topic)

            tag = kwargs.get('tag')
            if tag:
                self.msg.set_tags(tag)
        else:
            self.msg = None

    def send_sync(self, msg_body, **kwargs):
        """

        Args:
            msg_body:
            **kwargs:

        Returns:

        """
        topic = kwargs.get('topic')
        tag = kwargs.get('tag')
        assert self.msg is not None or topic is not None, \
            ValueError('当创建RocketMQ对象时没有设置topic，则发送消息时要指定topic和tag')
        assert self.msg is not None or tag is not None, \
            ValueError('当创建RocketMQ对象时没有设置topic，则发送消息时要指定topic和tag')

        if not self.msg:
            self.msg = Message(topic)
            self.msg.set_tags(tag)

        if isinstance(msg_body, str):
            try:
                _ = json.loads(msg_body)
            except Exception as e:
                print(traceback.format_exc())
                return 2, traceback.format_exc()
            msg_body.encode('utf8')
        else:
            msg_body = json.dumps(msg_body).encode('utf8')

        self.msg.set_body(msg_body)
        try:
            self.producer.send_sync(self.msg)
        except Exception as e:
            print(traceback.format_exc())
            return 1, traceback.format_exc()

        return 0, None
