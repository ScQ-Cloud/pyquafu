# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from requests.exceptions import RequestException

from ..exceptions import ServerError


class ClientWrapper:
    @staticmethod
    def post(*args, **kwargs):
        try:
            response = requests.post(*args, **kwargs)
            response.raise_for_status()
        except RequestException as err:
            raise ServerError(
                f"Failed to communicate with quafu website, please retry later or submit an issue, err: {err}"
            ) from err
        return response
