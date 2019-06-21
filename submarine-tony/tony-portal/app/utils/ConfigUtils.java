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

package utils;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigException;
import play.Logger;


/**
 * The class handles all configuration-related operations
 */
public class ConfigUtils {
  private static final Logger.ALogger LOG = Logger.of(ConfigUtils.class);

  /**
   * Check to see if configuration was passed in.
   * @param conf Config object. This is the wrapper of all args passed in from cmd line.
   * @return value of config if declared, {@code defaultVal} otherwise.
   */
  public static String fetchConfigIfExists(Config conf, String key, String defaultVal) {
    String value = defaultVal;
    try {
      value = conf.getString(key);
    } catch (ConfigException.Missing e) {
      LOG.warn("Failed to fetch value for `" + key + "`. Using `" + defaultVal + "` instead.", e);
    }
    return value;
  }

  public static int fetchIntConfigIfExists(Config conf, String key, int defaultVal) {
    if (conf.hasPath(key)) {
      return conf.getInt(key);
    }
    return defaultVal;
  }

  private ConfigUtils() { }
}
