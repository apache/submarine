package org.zapodot.junit.ldap;

import com.unboundid.ldap.sdk.schema.Schema;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class EmbeddedLdapRuleCustomWithoutSchemaTest {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder.newInstance()
                                                                      .withoutDefaultSchema()
                                                                      .build();

    @Test
    public void testEmptySchema() throws Exception {
        final Schema schema =
                embeddedLdapRule.ldapConnection().getSchema();
        assertTrue(schema.getAttributeTypes().isEmpty());

    }
}
