package org.zapodot.junit.ldap;

import com.unboundid.ldap.sdk.schema.AttributeTypeDefinition;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleCustomSchemaDuplicatedTest {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder.newInstance()
                                                                      .withSchema("standard-schema.ldif")
                                                                      .build();

    @Test
    public void testFindCustomAttribute() throws Exception {
        final AttributeTypeDefinition changelogAttribute =
                embeddedLdapRule.ldapConnection().getSchema().getAttributeType("changelog");
        assertNotNull(changelogAttribute);

    }
}
