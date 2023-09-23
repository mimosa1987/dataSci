# coding:utf8


class MySQLConf(object):
  """
    Mysql Configuration Bean
  """

  def __init__(self, host=None, port=3306, user=None, passwd=None, db_name=None, charset=None):
    """

    Args:
      host:
      port:
      user:
      passwd:
      db_name:
    """
    self._db_name = db_name
    self._passwd = passwd
    self._user = user
    self._port = port
    self._host = host
    self._charset = charset

  def set_conf_with_configparse(self, parser, section_name):
    """
    set properties with configuration
    Args:
      parser:
      section_name:

    Returns:

    """
    self._db_name = parser.get(section_name, 'db_name')
    self._passwd = parser.get(section_name, 'passwd')
    self._user = parser.get(section_name, 'user')
    self._port = int(parser.get(section_name, 'port'))
    self._host = parser.get(section_name, 'host')
    self._charset = parser.get(section_name, 'charset')

  @property
  def charset(self):
    return self._charset

  @charset.setter
  def charset(self, value):
    self._charset = value

  @property
  def host(self):
    return self._host

  @host.setter
  def host(self, value):
    self._host = value

  @property
  def port(self):
    return self._port

  @port.setter
  def port(self, value):
    self._port = value

  @property
  def user(self):
    return self._user

  @user.setter
  def user(self, value):
    self._user = value

  @property
  def passwd(self):
    return self._passwd

  @passwd.setter
  def passwd(self, value):
    self._passwd = value

  @property
  def db_name(self):
    return self._db_name

  @db_name.setter
  def db_name(self, value):
    self._db_name = value

  def to_dict(self):
    dict_ = {
      'host': self._host,
      'port': self._port,
      'user': self._user,
      'passwd': self._passwd,
      'db_name': self._db_name,
    }
    return dict_
