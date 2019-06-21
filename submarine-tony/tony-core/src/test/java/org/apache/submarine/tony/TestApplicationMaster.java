/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.tony;

import org.testng.Assert;
import org.testng.annotations.Test;


public class TestApplicationMaster {
  @Test
  public void testBuildBaseTaskCommand() {
    // null venv zip
    String actual = TonyClient.buildTaskCommand(null,
        "/export/apps/python/2.7/bin/python2.7",
        "src/main/python/my_awesome_script.py",
        "--input_dir hdfs://default/foo/bar");
    String expected = "/export/apps/python/2.7/bin/python2.7 "
                      + "src/main/python/my_awesome_script.py --input_dir hdfs://default/foo/bar";
    Assert.assertEquals(actual, expected);

    // venv zip is set, but should be ignored since pythonBinaryPath is absolute
    actual = TonyClient.buildTaskCommand(null,
        "/export/apps/python/2.7/bin/python2.7",
        "src/main/python/my_awesome_script.py",
        "--input_dir hdfs://default/foo/bar");
    expected = "/export/apps/python/2.7/bin/python2.7 "
               + "src/main/python/my_awesome_script.py --input_dir hdfs://default/foo/bar";
    Assert.assertEquals(actual, expected);

    // pythonBinaryPath is relative, so should be appended to "venv"
    actual = TonyClient.buildTaskCommand(null,
        "Python/bin/python",
        "src/main/python/my_awesome_script.py",
        "--input_dir hdfs://default/foo/bar");
    expected = "Python/bin/python "
               + "src/main/python/my_awesome_script.py --input_dir hdfs://default/foo/bar";
    Assert.assertEquals(actual, expected);

    // pythonBinaryPath is relative, so should be appended to "venv"
    actual = TonyClient.buildTaskCommand("hello", "Python/bin/python",
        "src/main/python/my_awesome_script.py", "--input_dir hdfs://default/foo/bar");
    expected = "venv/Python/bin/python "
        + "src/main/python/my_awesome_script.py --input_dir hdfs://default/foo/bar";
    Assert.assertEquals(actual, expected);
  }
}
