from ..exceptions import QuafuError


class UserError(QuafuError):
    pass


class APITokenNotFound(UserError):
    pass


class InvalidAPIToken(UserError):
    pass


class BackendNotAvailable(UserError):
    pass
