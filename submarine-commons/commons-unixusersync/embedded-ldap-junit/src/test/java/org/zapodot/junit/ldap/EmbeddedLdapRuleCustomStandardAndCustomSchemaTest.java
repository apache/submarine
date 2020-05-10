package org.zapodot.junit.ldap;

import com.unboundid.ldap.sdk.schema.AttributeTypeDefinition;
import com.unboundid.ldap.sdk.schema.Schema;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleCustomStandardAndCustomSchemaTest {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder.newInstance()
                                                                      .withSchema("custom-schema.ldif")
                                                                      .build();

    @Test
    public void testFindCustomAttribute() throws Exception {
        final Schema currentSchema = embeddedLdapRule.ldapConnection().getSchema();
        final AttributeTypeDefinition changelogAttribute =
                currentSchema.getAttributeType("attribute");
        assertNotNull(changelogAttribute);
        assertNotNull(currentSchema.getObjectClass("type"));
    }
}
