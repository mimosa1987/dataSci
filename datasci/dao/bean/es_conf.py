class ESConf():
  def __init__(self, host, port, user, passwd):
    """

    Args:
      host:
      port:
      user:
      passwd:
    """
    self._host = host
    self._user = user
    self._passwd = passwd
    self._port = port

  @property
  def host(self):
    return self._host

  @host.setter
  def host(self, value):
    self._host = value

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
  def port(self):
    return self._port

  @port.setter
  def port(self, value):
    self._port = value

  def to_dict(self):
    dict_ = {
      'host': self._host,
      'user': self._user,
      'passwd': self._passwd,
      'port': self._port,
    }
    return dict_
