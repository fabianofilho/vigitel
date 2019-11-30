from datapackage import Package

package = Package('datapackage.json')
package.get_resource('resource').read()

