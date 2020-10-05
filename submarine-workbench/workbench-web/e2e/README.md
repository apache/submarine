## Adding test cases

1. Create the test case file `*.e2e-spec.ts` as following code block under [here](./src/).

    describe("test case description", () => {

        beforeEach(() => {
            // set up actions before running test case
        });

        afterEach(() => {
            // set up actions after running test case
        });

        it("expected result description", function() {
            // expected result
            expect(expression).toEqual(true);
        });
    });

2. Add the test case file name `*.e2e-spec.ts` to the `specs` field of `protractor.conf.js`.

## Adding the browser for test cases

1. Add the browser to `travis.yml` for preparing testing environment.

        - name: Test submarine workbench-web Angular
        ...
        ...
        addons:
            chrome: stable
        script:
            - npm run test -- --no-watch --no-progress --browsers=ChromeHeadlessCI

2. Configure the browser into the `multiCapabilities` field of `protractor-ci.conf.js` for Angular.

        {
            browserName: 'chrome',
            chromeOptions: {
                args: ['--headless', '--no-sandbox']
            }
        }

3. Configure the browser into the `multiCapabilities` field of `protractor.conf.js` for Angular.

        {
            'browserName': 'chrome'
        }

4. Configure the browser launcher into `karma.conf.js` for the `karma` testing tool.  

        plugins: [
            ...
            require('karma-chrome-launcher'),
            ...
        ]
        ...
        browsers: ['Chrome'],
        ...
        customLaunchers: {
            ChromeHeadlessCI: {
                base: 'ChromeHeadless',
                flags: ['--no-sandbox']
            },
        },
        ...

5. Add the browser launcher into the `package.json` for the `npm` library configuration.    

        "devDependencies": {
            ...
            "karma-chrome-launcher": "~2.2.0",
            ...
        }


## Further helps

Click [here](https://angular.io/guide/testing) for further information.