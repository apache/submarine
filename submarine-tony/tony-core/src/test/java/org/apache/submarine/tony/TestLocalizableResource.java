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

import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TestLocalizableResource {
  @Test
  public void testLocalResourceParsing() throws IOException, ParseException {
    FileSystem fs = FileSystem.get(new Configuration());
    LocalizableResource resource
        = new LocalizableResource("src/test/resources/test.zip", fs);
    Assert.assertNotNull(resource.toLocalResource().getResource());
    Assert.assertEquals(resource.getLocalizedFileName(), "test.zip");

    LocalizableResource resource2
        = new LocalizableResource("src/test/resources/test.zip::ok.123", fs);
    Assert.assertNotNull(resource2.toLocalResource().getResource());
    Assert.assertEquals(resource2.getLocalizedFileName(), "ok.123");

    LocalizableResource resource3
        = new LocalizableResource("src/test/resources/test.zip::ok#archive", fs);
    Assert.assertNotNull(resource3.toLocalResource().getResource());
    Assert.assertSame(resource3.toLocalResource().getType(), LocalResourceType.ARCHIVE);
    Assert.assertEquals(resource3.getLocalizedFileName(), "ok");

    LocalizableResource resource4
        = new LocalizableResource("src/test/resources/test.zip#archive", fs);
    Assert.assertNotNull(resource4.toLocalResource().getResource());
    Assert.assertSame(resource4.toLocalResource().getType(), LocalResourceType.ARCHIVE);
    Assert.assertEquals(resource4.getLocalizedFileName(), "test.zip");
  }

}
