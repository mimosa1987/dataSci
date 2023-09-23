# coding:utf8

from elasticsearch.helpers import scan
from elasticsearch import Elasticsearch
from ..dao.bean.es_conf import ESConf
from ..constant import VALUE_TYPE_ERROR_TIPS
from elasticsearch import helpers
from ..dao import Dao


class ESDao(Dao):
    """
      ElasticSearch data access object.
    """

    def __init__(self, conf=None, auto_connect=True):
        """

        Args:
          conf: Configuration for the ES.
          auto_connect:
        """
        super(ESDao, self).__init__(conf)

        self.connector = None
        assert isinstance(conf, ESConf), ValueError(VALUE_TYPE_ERROR_TIPS)

        self._conf = conf

        if auto_connect:
            self.connector = Elasticsearch([conf.host], http_auth=(conf.user, conf.passwd), port=conf.port)

    def connect(self):
        """

        Returns:

        """
        conf = self.conf
        self.connector = Elasticsearch([conf.host], http_auth=(conf.user, conf.passwd), port=conf.port)

    def disconnect(self):
        """

        Returns:

        """
        pass

    def bulk(self, actions):
        """

        Args:
          actions:

        Returns:

        """
        helpers.bulk(self.connector, actions)

    def index(self, index, doc_id, doc_type, body):
        """

        Args:
          index:
          doc_type:
          body:

        Returns:

        """
        self.connector.index(index=index, doc_type=doc_type, id=doc_id, body=body)

    def query(self, index_name, condition_attr_names=None, condition_attr_values=None, condition_type='match_phrase',
              udc=None, request_timeout=None, raise_on_error=False):
        """
        Query the data according to the given conditions
        Args:
            index_name:  str value, es index name
            condition_attr_names:  list value, condition attributes
            condition_attr_values:  list value, condition value for given condition attributes
            condition_type:  str value, condition's type, default 'match_phrase'
            udc: dict value, user defined condition
            request_timeout:  int value, max time waiting
            raise_on_error:  whether raise exception if error, default False

        Returns:
            eligible data
        """
        must_conditions = list()

        if udc is not None:
            q_str = {
                "query": udc
            }
        else:
            for name, value in zip(condition_attr_names, condition_attr_values):
                must_conditions.append({condition_type: {name: value}})
            q_str = {
                "query": {
                    "bool": {
                        "must": must_conditions
                    }
                }
            }

        result = scan(self.connector, query=q_str, index=index_name,
                      request_timeout=request_timeout, raise_on_error=raise_on_error)

        return [item['_source'] for item in result]

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, value):
        self._conf = value
