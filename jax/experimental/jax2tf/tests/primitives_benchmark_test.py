# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest

from jax import test_util as jtu
from jax.config import config
from jax.experimental.jax2tf.tests import primitives_test

config.parse_flags_with_absl()

from typing import Callable

class JaxPrimitiveBenchmarkTest(primitives_test.JaxPrimitiveTest):
  def ConvertAndCompare(self, func_jax: Callable, *args,
                        enable_xla: bool = True, **kwargs):
    self.ConvertAndBenchmark(func_jax, *args, enable_xla=enable_xla)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
