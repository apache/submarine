package org.apache.submarine.commons.unixusersync;


import javax.crypto.spec.SecretKeySpec;
import javax.xml.bind.DatatypeConverter;
import java.security.Key;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtBuilder;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.impl.crypto.MacProvider;

import java.util.Base64;
import java.util.Date;

public class CreateToken {

  private static final Key secret = MacProvider.generateKey(SignatureAlgorithm.HS256);
  private static final byte[] secretBytes = secret.getEncoded();
  private static final String base64SecretBytes = Base64.getEncoder().encodeToString(secretBytes);

  //Sample method to construct a JWT

  public static String createJWT(String id, String issuer, String subject, long ttlMillis) {
    //The JWT signature algorithm we will be using to sign the token
    SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.HS256;

    long nowMillis = System.currentTimeMillis();
    Date now = new Date(nowMillis);

    //We will sign our JWT with our ApiKey secret
    Key signingKey = new SecretKeySpec(secretBytes, signatureAlgorithm.getJcaName());

    //Let's set the JWT Claims
    JwtBuilder builder = Jwts.builder().setId(id)
            .setIssuedAt(now)
            .setSubject(subject)
            .setIssuer(issuer)
            .signWith(signatureAlgorithm, signingKey);

    //if it has been specified, let's add the expiration
    if (ttlMillis >= 0) {
      long expMillis = nowMillis + ttlMillis;
      Date exp = new Date(expMillis);
      builder.setExpiration(exp);
    }

    //Builds the JWT and serializes it to a compact, URL-safe string
    return builder.compact();
  }

  //Sample method to validate and read the JWT
  public void parseJWT(String jwt) {
    //This line will throw an exception if it is not a signed JWS (as expected)
    Claims claims = Jwts.parser()
        .setSigningKey(DatatypeConverter.parseBase64Binary(base64SecretBytes))
        .parseClaimsJws(jwt).getBody();
    System.out.println("ID: " + claims.getId());
    System.out.println("Subject: " + claims.getSubject());
    System.out.println("Issuer: " + claims.getIssuer());
    System.out.println("Expiration: " + claims.getExpiration());
  }
}
