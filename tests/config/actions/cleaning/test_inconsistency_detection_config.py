from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    PanderaSchemaValidator,
)
from learn2clean.configs.actions.cleaning.inconsistency_detection import (
    PanderaSchemaValidatorConfig,
)


class TestInconsistencyDetectionConfigs:
    def test_pandera_validator_default_config(self):
        cfg = OmegaConf.structured(PanderaSchemaValidatorConfig)
        action = instantiate(cfg)
        assert isinstance(action, PanderaSchemaValidator)
        assert action.name == "PanderaSchemaValidator"
        assert action.params.get("remove_inconsistent_rows") is True
        default_schema = action.params.get("schema_config")
        assert "columns" in default_schema

    def test_pandera_validator_custom_schema(self):
        complex_schema = {
            "columns": {
                "age": {
                    "dtype": "int",
                    "checks": [{"greater_than_or_equal_to": 18}, {"less_than": 120}],
                },
                "email": {
                    "dtype": "str",
                    "checks": [{"str_matches": r"^[\w\.-]+@[\w\.-]+\.\w+$"}],
                },
            }
        }
        cfg = OmegaConf.structured(
            PanderaSchemaValidatorConfig(
                schema_config=complex_schema,
                remove_inconsistent_rows=False,
                name="MyCustomValidator",
            )
        )
        action = instantiate(cfg)
        assert action.name == "MyCustomValidator"
        assert action.params.get("remove_inconsistent_rows") is False
        injected_schema = action.params.get("schema_config")
        assert injected_schema["columns"]["age"]["dtype"] == "int"
        assert len(injected_schema["columns"]["age"]["checks"]) == 2
        assert injected_schema["columns"]["email"]["dtype"] == "str"

    def test_pandera_inheritance(self):
        cfg = OmegaConf.structured(
            PanderaSchemaValidatorConfig(columns=["age", "email"], exclude=["id"])
        )
        action = instantiate(cfg)
        assert action.columns == ["age", "email"]
        assert action.exclude == ["id"]
