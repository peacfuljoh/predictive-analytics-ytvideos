"""ETL request parent class"""



class ETLRequest():
    """Parent class for ETL requests."""
    def __init__(self,
                 config: dict,
                 name: str):
        # fields
        self.name = name
        self._extract: dict = None
        self._transform: dict = None
        self._load: dict = None
        self._valid = False

        # set fields with validation
        config = self._validate_config(config)
        self._extract = config['extract']
        self._transform = config['transform']
        self._load = config['load']

    def _validate_config(self, config: dict) -> dict:
        """Validate configuration"""
        for key in ['extract', 'transform', 'load']:
            if key not in config:
                config[key] = {}
        self._validate_config_extract(config['extract'])
        self._validate_config_transform(config['transform'])
        self._validate_config_load(config['load'])
        return config

    def _validate_config_extract(self, config: dict):
        """Validate extract configuration"""
        raise NotImplementedError

    def _validate_config_transform(self, config: dict):
        """Validate transform configuration"""
        raise NotImplementedError

    def _validate_config_load(self, config: dict):
        """Validate load configuration"""
        raise NotImplementedError

    def _validate_config_keys(self,
                              sub_config: dict,
                              mode: str):
        """Validate keys specified in a sub-config."""
        raise NotImplementedError

    def _check_if_valid(self):
        """Check if configuration is valid"""
        if not self._valid:
            raise Exception('ETL request has not been validated.')

    def get_config_as_dict(self) -> dict:
        raise NotImplementedError

    def get_extract(self) -> dict:
        """Get extract configuration"""
        self._check_if_valid()
        return self._extract

    def get_transform(self) -> dict:
        """Get transform configuration"""
        self._check_if_valid()
        return self._transform

    def get_load(self) -> dict:
        """Get load configuration"""
        self._check_if_valid()
        return self._load

    def set_valid(self, valid: bool):
        """Set config validity"""
        self._valid = valid
