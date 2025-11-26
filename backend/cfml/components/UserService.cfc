<cfcomponent displayname="UserService">

    <cffunction name="getUserInfo" access="remote" returntype="struct">
        <cfargument name="id" type="numeric" required="true">

        <cfset var result = {}>

        <cfset result = {
            "id": arguments.id,
            "status": "active",
            "message": "CFML endpoint is alive"
        }>

        <cfreturn result>
    </cffunction>

</cfcomponent>
